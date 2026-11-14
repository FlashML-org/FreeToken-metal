// mtlstat: a DYLD_INSERT_LIBRARIES hook that counts Metal compute dispatches and command buffer
// commits, accumulates the GPU time of completed command buffers and, optionally, logs the kernels
// of every command buffer. Works for any Metal process (MLX, torch-MPS, llama.cpp).
//
// Every MTLSTAT_FLUSH_MS (default 20) and once more at exit it appends the cumulative counters to
// $MTLSTAT_DIR/mtlstat.<pid>.tsv (default dir /tmp; a directory it cannot write to gets one error
// on stderr and nothing is recorded):
//     t_host_s  dispatches  commits  cb_done  gpu_busy_s  gpu_sum_s
// (busy = union of the command buffers' GPU intervals, sum = their plain sum; t_host_s and the GPU
// times share the mach_absolute_time clock). With MTLSTAT_CBLOG=1 every committed command buffer
// is also logged on completion to $MTLSTAT_DIR/cblog.<pid>.tsv as
//     gpu_start_s  gpu_end_s  n_dispatch  kernel,kernel,...
// Kernel names come from hooking pipeline creation (MTLFunction.name -> associated object on the
// pipeline), setComputePipelineState: (pipeline -> encoder) and encoder creation (encoder ->
// command buffer). With one kernel per command buffer (MLX_MAX_OPS_PER_BUFFER=0), gpu_end -
// gpu_start is that kernel's GPU time plus a small per-buffer overhead. The exit-time flush first
// waits, up to 1 s, for the completion handlers of committed buffers; os._exit() skips it, so a
// workload should end with mx.synchronize() and a normal exit. MTLSTAT_VERBOSE (any value) prints
// the pid and the number of hooked methods to stderr at load.
//
// Build: cc -O2 -fobjc-arc -dynamiclib -framework Foundation -framework Metal mtlstat.m
//        -o libmtlstat.dylib   (tools/profiling/build.sh)

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <objc/runtime.h>
#include <errno.h>
#include <mach/mach_time.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define MAX_HOOKS 64
#define DEFAULT_FLUSH_MS 20
#define MAX_PATH_LENGTH 1024
#define EXIT_WAIT_MS 1000
// Completion handlers do not run in GPU start order, so the busy union keeps the latest disjoint
// intervals; one that arrives older than all of them is counted without overlap.
#define RECENT_INTERVALS 128

static _Atomic unsigned long long g_dispatches = 0;
static _Atomic unsigned long long g_commits = 0;
static _Atomic unsigned long long g_completed = 0;
static int g_log_enabled = 0;
static FILE* g_counter_file;
static FILE* g_log_file;
// Held for a whole flush, so the exit flush waits for one in progress and lines stay in order.
static pthread_mutex_t g_write_mutex = PTHREAD_MUTEX_INITIALIZER;

// Guarded by g_mutex.
static pthread_mutex_t g_mutex = PTHREAD_MUTEX_INITIALIZER;
static double g_busy_s = 0;
static double g_sum_s = 0;
static struct {
  double start_s;
  double end_s;
} g_recent[RECENT_INTERVALS]; // disjoint, sorted by start
static int g_recent_count = 0;
static NSMutableString* g_log_lines;

// Associated object keys: their addresses are the keys.
static char kPipelineName;
static char kEncoderKernel;
static char kEncoderCommandBuffer;
static char kCommandBufferKernels;

static double now_s(void) {
  static mach_timebase_info_data_t timebase;
  if (timebase.denom == 0) {
    mach_timebase_info(&timebase);
  }
  return (double)mach_absolute_time() * timebase.numer / timebase.denom / 1e9;
}

// The replaced methods and their original implementations, which the hooks call through to.
static struct {
  Method method;
  IMP original;
  SEL selector;
} g_hooks[MAX_HOOKS];
static int g_hook_count = 0;

static IMP original_imp(id self, SEL selector) {
  Method method = class_getInstanceMethod(object_getClass(self), selector);
  for (int i = 0; i < g_hook_count; i++) {
    if (g_hooks[i].method == method) {
      return g_hooks[i].original;
    }
  }
  // A subclass override that calls super reaches the hook through its own Method.
  for (int i = 0; i < g_hook_count; i++) {
    if (g_hooks[i].selector == selector) {
      return g_hooks[i].original;
    }
  }
  return NULL;
}

static void hook(Class cls, SEL selector, IMP replacement) {
  if (!cls) {
    return;
  }
  Method method = class_getInstanceMethod(cls, selector);
  if (!method || g_hook_count >= MAX_HOOKS) {
    return;
  }
  for (int i = 0; i < g_hook_count; i++) {
    if (g_hooks[i].method == method) {
      return;
    }
  }
  g_hooks[g_hook_count].method = method;
  g_hooks[g_hook_count].original = method_getImplementation(method);
  g_hooks[g_hook_count].selector = selector;
  g_hook_count++;
  method_setImplementation(method, replacement);
}

static void name_pipeline(id pipeline, NSString* name) {
  if (pipeline && name) {
    objc_setAssociatedObject(pipeline, &kPipelineName, name, OBJC_ASSOCIATION_RETAIN_NONATOMIC);
  }
}

typedef id (*NewPipelineWithFunction)(id, SEL, id, NSError**);
typedef id (*NewPipelineWithOptions)(id, SEL, id, NSUInteger, id*, NSError**);

static id hooked_new_pipeline_with_function(id self, SEL _cmd, id function, NSError** error) {
  id pipeline = ((NewPipelineWithFunction)original_imp(self, _cmd))(self, _cmd, function, error);
  name_pipeline(pipeline, [(id<MTLFunction>)function name]);
  return pipeline;
}

static id hooked_new_pipeline_with_function_options(
    id self,
    SEL _cmd,
    id function,
    NSUInteger options,
    id* reflection,
    NSError** error) {
  id pipeline = ((NewPipelineWithOptions)original_imp(self, _cmd))(
      self, _cmd, function, options, reflection, error);
  name_pipeline(pipeline, [(id<MTLFunction>)function name]);
  return pipeline;
}

static id hooked_new_pipeline_with_descriptor(
    id self,
    SEL _cmd,
    id descriptor,
    NSUInteger options,
    id* reflection,
    NSError** error) {
  id pipeline = ((NewPipelineWithOptions)original_imp(self, _cmd))(
      self, _cmd, descriptor, options, reflection, error);
  MTLComputePipelineDescriptor* pipeline_descriptor = descriptor;
  name_pipeline(pipeline, pipeline_descriptor.label ?: [pipeline_descriptor.computeFunction name]);
  return pipeline;
}

typedef id (*NewEncoder)(id, SEL);
typedef id (*NewEncoderWithType)(id, SEL, NSUInteger);
typedef id (*NewEncoderWithDescriptor)(id, SEL, id);

static id tag_encoder(id command_buffer, id encoder) {
  if (encoder) {
    objc_setAssociatedObject(
        encoder, &kEncoderCommandBuffer, command_buffer, OBJC_ASSOCIATION_ASSIGN);
  }
  return encoder;
}

static id hooked_new_encoder(id self, SEL _cmd) {
  return tag_encoder(self, ((NewEncoder)original_imp(self, _cmd))(self, _cmd));
}

static id hooked_new_encoder_with_type(id self, SEL _cmd, NSUInteger dispatch_type) {
  return tag_encoder(
      self, ((NewEncoderWithType)original_imp(self, _cmd))(self, _cmd, dispatch_type));
}

static id hooked_new_encoder_with_descriptor(id self, SEL _cmd, id descriptor) {
  return tag_encoder(
      self, ((NewEncoderWithDescriptor)original_imp(self, _cmd))(self, _cmd, descriptor));
}

typedef void (*SetPipeline)(id, SEL, id);

static void hooked_set_pipeline(id self, SEL _cmd, id pipeline) {
  if (g_log_enabled) {
    NSString* name = objc_getAssociatedObject(pipeline, &kPipelineName);
    objc_setAssociatedObject(
        self, &kEncoderKernel, name ?: @"?", OBJC_ASSOCIATION_RETAIN_NONATOMIC);
  }
  ((SetPipeline)original_imp(self, _cmd))(self, _cmd, pipeline);
}

static void count_dispatch(id encoder) {
  atomic_fetch_add(&g_dispatches, 1);
  if (!g_log_enabled) {
    return;
  }
  id command_buffer = objc_getAssociatedObject(encoder, &kEncoderCommandBuffer);
  if (!command_buffer) {
    return;
  }
  NSMutableArray* kernels = objc_getAssociatedObject(command_buffer, &kCommandBufferKernels);
  if (!kernels) {
    kernels = [NSMutableArray array];
    objc_setAssociatedObject(
        command_buffer, &kCommandBufferKernels, kernels, OBJC_ASSOCIATION_RETAIN_NONATOMIC);
  }
  [kernels addObject:objc_getAssociatedObject(encoder, &kEncoderKernel) ?: @"?"];
}

typedef void (*Dispatch)(id, SEL, MTLSize, MTLSize);
typedef void (*DispatchIndirect)(id, SEL, id, NSUInteger, MTLSize);

static void hooked_dispatch(id self, SEL _cmd, MTLSize size, MTLSize threadgroup_size) {
  count_dispatch(self);
  ((Dispatch)original_imp(self, _cmd))(self, _cmd, size, threadgroup_size);
}

static void hooked_dispatch_indirect(
    id self,
    SEL _cmd,
    id buffer,
    NSUInteger offset,
    MTLSize threadgroup_size) {
  count_dispatch(self);
  ((DispatchIndirect)original_imp(self, _cmd))(self, _cmd, buffer, offset, threadgroup_size);
}

// Merges [start_s, end_s] into g_recent and returns the time it adds to the union. Caller holds
// g_mutex.
static double add_busy_interval(double start_s, double end_s) {
  double added_s = end_s - start_s;
  int first = -1;
  int last = -1;
  for (int i = 0; i < g_recent_count; i++) {
    if (g_recent[i].end_s >= start_s && g_recent[i].start_s <= end_s) {
      double from_s = start_s > g_recent[i].start_s ? start_s : g_recent[i].start_s;
      double to_s = end_s < g_recent[i].end_s ? end_s : g_recent[i].end_s;
      added_s -= to_s > from_s ? to_s - from_s : 0;
      first = first < 0 ? i : first;
      last = i;
    }
  }
  if (first >= 0) {
    g_recent[first].start_s = start_s < g_recent[first].start_s ? start_s : g_recent[first].start_s;
    g_recent[first].end_s = end_s > g_recent[last].end_s ? end_s : g_recent[last].end_s;
    memmove(
        &g_recent[first + 1],
        &g_recent[last + 1],
        (g_recent_count - last - 1) * sizeof g_recent[0]);
    g_recent_count -= last - first;
    return added_s;
  }
  int at = 0;
  while (at < g_recent_count && g_recent[at].start_s < start_s) {
    at++;
  }
  if (g_recent_count == RECENT_INTERVALS && at == 0) {
    return added_s;
  } else if (g_recent_count == RECENT_INTERVALS) {
    memmove(&g_recent[0], &g_recent[1], (at - 1) * sizeof g_recent[0]);
    at--;
  } else {
    memmove(&g_recent[at + 1], &g_recent[at], (g_recent_count - at) * sizeof g_recent[0]);
    g_recent_count++;
  }
  g_recent[at].start_s = start_s;
  g_recent[at].end_s = end_s;
  return added_s;
}

typedef void (*Commit)(id, SEL);

static void hooked_commit(id self, SEL _cmd) {
  atomic_fetch_add(&g_commits, 1);
  NSArray* kernels =
      g_log_enabled ? [objc_getAssociatedObject(self, &kCommandBufferKernels) copy] : nil;
  [(id<MTLCommandBuffer>)self addCompletedHandler:^(id<MTLCommandBuffer> command_buffer) {
    double start_s = command_buffer.GPUStartTime;
    double end_s = command_buffer.GPUEndTime;
    atomic_fetch_add(&g_completed, 1);
    pthread_mutex_lock(&g_mutex);
    if (end_s > start_s && start_s > 0) {
      g_sum_s += end_s - start_s;
      g_busy_s += add_busy_interval(start_s, end_s);
    }
    if (g_log_enabled) {
      [g_log_lines appendFormat:@"%.9f\t%.9f\t%lu\t%@\n",
                                start_s,
                                end_s,
                                (unsigned long)kernels.count,
                                [kernels componentsJoinedByString:@","]];
    }
    pthread_mutex_unlock(&g_mutex);
  }];
  ((Commit)original_imp(self, _cmd))(self, _cmd);
}

static void flush_once(void) {
  pthread_mutex_lock(&g_write_mutex);
  @autoreleasepool {
    NSString* lines = nil;
    pthread_mutex_lock(&g_mutex);
    double busy_s = g_busy_s;
    double sum_s = g_sum_s;
    if (g_log_file && g_log_lines.length) {
      lines = [g_log_lines copy];
      [g_log_lines setString:@""];
    }
    pthread_mutex_unlock(&g_mutex);
    if (g_counter_file) {
      fprintf(
          g_counter_file,
          "%.6f\t%llu\t%llu\t%llu\t%.6f\t%.6f\n",
          now_s(),
          atomic_load(&g_dispatches),
          atomic_load(&g_commits),
          atomic_load(&g_completed),
          busy_s,
          sum_s);
      fflush(g_counter_file);
    }
    if (lines && g_log_file) {
      fputs(lines.UTF8String, g_log_file);
      fflush(g_log_file);
    }
  }
  pthread_mutex_unlock(&g_write_mutex);
}

// $MTLSTAT_DIR/<name>.<pid>.tsv opened for writing, or NULL after an error on stderr.
static FILE* open_output(const char* name) {
  char path[MAX_PATH_LENGTH];
  snprintf(path, sizeof path, "%s/%s.%d.tsv", getenv("MTLSTAT_DIR") ?: "/tmp", name, getpid());
  FILE* file = fopen(path, "w");
  if (!file) {
    fprintf(stderr, "[mtlstat] pid %d: cannot write %s: %s\n", getpid(), path, strerror(errno));
  }
  return file;
}

// pthread_create's start routine signature; the argument is unused.
static void* flush_periodically(void* unused) {
  const char* flush_ms_text = getenv("MTLSTAT_FLUSH_MS");
  int flush_ms = flush_ms_text ? atoi(flush_ms_text) : DEFAULT_FLUSH_MS;
  if (flush_ms < 1) {
    flush_ms = 1;
  }
  for (;;) {
    flush_once();
    usleep(flush_ms * 1000);
  }
  return NULL;
}

__attribute__((destructor)) static void mtlstat_fini(void) {
  // Completion handlers run on Metal's threads and can trail the wait that let the process exit.
  for (int waited_ms = 0;
       waited_ms < EXIT_WAIT_MS && atomic_load(&g_completed) < atomic_load(&g_commits);
       waited_ms++) {
    usleep(1000);
  }
  flush_once();
}

__attribute__((constructor)) static void mtlstat_init(void) {
  @autoreleasepool {
    const char* log_text = getenv("MTLSTAT_CBLOG");
    g_log_enabled = log_text && atoi(log_text);
    g_log_lines = [NSMutableString new];
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
      return;
    }
    g_counter_file = open_output("mtlstat");
    if (!g_counter_file) {
      return;
    }
    fprintf(g_counter_file, "t_host_s\tdispatches\tcommits\tcb_done\tgpu_busy_s\tgpu_sum_s\n");
    if (g_log_enabled) {
      g_log_file = open_output("cblog");
      g_log_enabled = g_log_file != NULL;
    }
    Class device_class = object_getClass(device);
    hook(
        device_class,
        @selector(newComputePipelineStateWithFunction:error:),
        (IMP)hooked_new_pipeline_with_function);
    hook(
        device_class,
        @selector(newComputePipelineStateWithFunction:options:reflection:error:),
        (IMP)hooked_new_pipeline_with_function_options);
    hook(
        device_class,
        @selector(newComputePipelineStateWithDescriptor:options:reflection:error:),
        (IMP)hooked_new_pipeline_with_descriptor);
    // The command buffer and encoder classes are private: make one of each kind to find them.
    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLCommandBuffer> retained_buffer = [queue commandBuffer];
    id<MTLCommandBuffer> unretained_buffer = [queue commandBufferWithUnretainedReferences];
    id<MTLComputeCommandEncoder> serial_encoder = [retained_buffer computeCommandEncoder];
    Class serial_encoder_class = object_getClass(serial_encoder);
    [serial_encoder endEncoding];
    id<MTLComputeCommandEncoder> concurrent_encoder =
        [unretained_buffer computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
    Class concurrent_encoder_class = object_getClass(concurrent_encoder);
    [concurrent_encoder endEncoding];
    Class buffer_classes[] = {object_getClass(retained_buffer), object_getClass(unretained_buffer)};
    Class encoder_classes[] = {serial_encoder_class, concurrent_encoder_class};
    for (int i = 0; i < 2; i++) {
      hook(
          encoder_classes[i],
          @selector(dispatchThreadgroups:threadsPerThreadgroup:),
          (IMP)hooked_dispatch);
      hook(
          encoder_classes[i],
          @selector(dispatchThreads:threadsPerThreadgroup:),
          (IMP)hooked_dispatch);
      hook(
          encoder_classes[i],
          @selector(
              dispatchThreadgroupsWithIndirectBuffer:indirectBufferOffset:threadsPerThreadgroup:),
          (IMP)hooked_dispatch_indirect);
      hook(encoder_classes[i], @selector(setComputePipelineState:), (IMP)hooked_set_pipeline);
      hook(buffer_classes[i], @selector(computeCommandEncoder), (IMP)hooked_new_encoder);
      hook(
          buffer_classes[i],
          @selector(computeCommandEncoderWithDispatchType:),
          (IMP)hooked_new_encoder_with_type);
      hook(
          buffer_classes[i],
          @selector(computeCommandEncoderWithDescriptor:),
          (IMP)hooked_new_encoder_with_descriptor);
      hook(buffer_classes[i], @selector(commit), (IMP)hooked_commit);
    }
    if (getenv("MTLSTAT_VERBOSE")) {
      fprintf(
          stderr,
          "[mtlstat] pid %d hooked %d methods (cblog=%d)\n",
          getpid(),
          g_hook_count,
          g_log_enabled);
    }
    pthread_t thread;
    pthread_create(&thread, NULL, flush_periodically, NULL);
    pthread_detach(thread);
  }
}
