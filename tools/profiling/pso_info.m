// pso_info: compile a Metal library source (or load a .metallib) and print, for one or all
// kernels, the pipeline properties MLX does not expose:
//   maxTotalThreadsPerThreadgroup  below 1024, and below the N of a kernel that declares
//                                  [[max_total_threads_per_threadgroup(N)]], the compiler capped
//                                  the threadgroup because of register use
//   threadExecutionWidth           SIMD width, 32 on Apple GPUs
//   staticThreadgroupMemoryLength  threadgroup memory declared statically in the kernel
// Sources compile as MLX compiles them: Metal 4.0, safe math (MLX's default math mode). They are
// prepared as ftmetal prepares a C++ primitive's library: each #include "x" is inlined once from
// the first -I dir that has it and #pragma once dropped (python/ftmetal/kernel/_ext.py), then
// every {{KEY}} is replaced (the primitive's .cpp; here --sub KEY=VAL).
//
// Build: cc -O2 -fobjc-arc -framework Foundation -framework Metal pso_info.m -o pso_info
//        (tools/profiling/build.sh)

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <string.h>

static const char* USAGE =
    "usage: pso_info <file.metal|file.metallib> [kernel] [-I dir ...] [-D NAME=VAL ...]\n"
    "                [--sub KEY=VAL ...]\n"
    "Compiles the source as MLX does (Metal 4.0, safe math) and prints the kernel's (or every\n"
    "kernel's) maxTotalThreadsPerThreadgroup, threadExecutionWidth and\n"
    "staticThreadgroupMemoryLength.\n"
    "  -I dir        inline each #include \"x\" once from the first dir that has it; for ftmetal,\n"
    "                -I python/ftmetal/kernel -I .venv/lib/python3.13/site-packages/mlx/include\n"
    "  --sub KEY=VAL replace every {{KEY}} after inlining; a {{KEY}} left over is an error\n"
    "  -D NAME=VAL   preprocessor macro (VAL defaults to 1)\n"
    "Exits 1 when a kernel is missing or a pipeline is not built. Limits: function constants are\n"
    "not set, so a kernel that requires them is listed as such, not built; an\n"
    "mx.fast.metal_kernel body (a .metal without [[kernel]] functions) does not compile.\n";

// KEY=VALUE adds {KEY: VALUE} to the pairs; a bare KEY maps to missing_value.
static void add_pair(NSMutableDictionary* pairs, const char* text, NSString* missing_value) {
  NSString* pair = @(text);
  NSRange equals = [pair rangeOfString:@"="];
  if (equals.location == NSNotFound) {
    pairs[pair] = missing_value;
  } else {
    pairs[[pair substringToIndex:equals.location]] = [pair substringFromIndex:equals.location + 1];
  }
}

// The source with each #include "x" replaced by x, depth first and the first time only, and every
// #pragma once dropped; nil after printing the include that no -I dir has.
static NSString* inline_headers(
    NSString* source,
    NSArray<NSString*>* include_dirs,
    NSMutableSet<NSString*>* included) {
  NSRegularExpression* pattern =
      [NSRegularExpression regularExpressionWithPattern:@"^#include \"([^\"]+)\"$|^#pragma once$"
                                                options:NSRegularExpressionAnchorsMatchLines
                                                  error:nil];
  NSMutableString* result = [NSMutableString new];
  NSUInteger copied = 0;
  for (NSTextCheckingResult* match in [pattern matchesInString:source
                                                       options:0
                                                         range:NSMakeRange(0, source.length)]) {
    [result appendString:[source substringWithRange:NSMakeRange(
                                                        copied, match.range.location - copied)]];
    copied = NSMaxRange(match.range);
    NSRange header_range = [match rangeAtIndex:1];
    if (header_range.location == NSNotFound) {
      continue;
    }
    NSString* header = [source substringWithRange:header_range];
    if ([included containsObject:header]) {
      continue;
    }
    [included addObject:header];
    NSString* header_source = nil;
    for (NSString* dir in include_dirs) {
      header_source = [NSString stringWithContentsOfFile:[dir stringByAppendingPathComponent:header]
                                                encoding:NSUTF8StringEncoding
                                                   error:nil];
      if (header_source) {
        break;
      }
    }
    if (!header_source) {
      fprintf(stderr, "cannot inline #include \"%s\": no -I dir has it\n", header.UTF8String);
      return nil;
    }
    NSString* inlined = inline_headers(header_source, include_dirs, included);
    if (!inlined) {
      return nil;
    }
    [result appendString:inlined];
  }
  [result appendString:[source substringFromIndex:copied]];
  return result;
}

int main(int argc, char** argv) {
  @autoreleasepool {
    if (argc < 2) {
      fprintf(stderr, "%s", USAGE);
      return 2;
    }
    if (!strcmp(argv[1], "-h") || !strcmp(argv[1], "--help")) {
      printf("%s", USAGE);
      return 0;
    }
    NSString* path = @(argv[1]);
    NSString* kernel_name = nil;
    NSMutableDictionary* macros = [NSMutableDictionary new];
    NSMutableArray* include_dirs = [NSMutableArray new];
    NSMutableDictionary* substitutions = [NSMutableDictionary new];
    for (int i = 2; i < argc; i++) {
      if (!strcmp(argv[i], "--sub") && i + 1 < argc) {
        add_pair(substitutions, argv[++i], @"");
      } else if (!strcmp(argv[i], "-I") && i + 1 < argc) {
        [include_dirs addObject:@(argv[++i])];
      } else if (!strncmp(argv[i], "-I", 2) && argv[i][2]) {
        [include_dirs addObject:@(argv[i] + 2)];
      } else if (!strcmp(argv[i], "-D") && i + 1 < argc) {
        add_pair(macros, argv[++i], @"1");
      } else if (argv[i][0] == '-' || kernel_name) {
        fprintf(stderr, "unexpected argument %s\n%s", argv[i], USAGE);
        return 2;
      } else {
        kernel_name = @(argv[i]);
      }
    }

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    NSError* error = nil;
    id<MTLLibrary> library = nil;
    if ([path hasSuffix:@".metallib"]) {
      library = [device newLibraryWithURL:[NSURL fileURLWithPath:path] error:&error];
    } else {
      NSString* source = [NSString stringWithContentsOfFile:path
                                                   encoding:NSUTF8StringEncoding
                                                      error:&error];
      if (!source) {
        fprintf(stderr, "read failed: %s\n", error.localizedDescription.UTF8String);
        return 1;
      }
      source = inline_headers(source, include_dirs, [NSMutableSet new]);
      if (!source) {
        return 1;
      }
      for (NSString* key in substitutions) {
        source =
            [source stringByReplacingOccurrencesOfString:[NSString stringWithFormat:@"{{%@}}", key]
                                              withString:substitutions[key]];
      }
      NSRegularExpression* marker_pattern =
          [NSRegularExpression regularExpressionWithPattern:@"\\{\\{[A-Z][A-Z0-9_]*\\}\\}"
                                                    options:0
                                                      error:nil];
      NSMutableOrderedSet* markers = [NSMutableOrderedSet new];
      for (NSTextCheckingResult* match in
           [marker_pattern matchesInString:source options:0 range:NSMakeRange(0, source.length)]) {
        [markers addObject:[source substringWithRange:match.range]];
      }
      if (markers.count) {
        fprintf(
            stderr,
            "unsubstituted markers, give each with --sub KEY=VAL: %s\n",
            [markers.array componentsJoinedByString:@" "].UTF8String);
        return 1;
      }
      MTLCompileOptions* options = [MTLCompileOptions new];
      options.languageVersion = MTLLanguageVersion4_0;
      options.mathMode = MTLMathModeSafe;
      options.preprocessorMacros = macros;
      library = [device newLibraryWithSource:source options:options error:&error];
    }
    if (!library) {
      fprintf(stderr, "compile failed:\n%s\n", error.localizedDescription.UTF8String);
      return 1;
    }

    NSMutableArray<id<MTLFunction>>* kernels = [NSMutableArray new];
    for (NSString* name in library.functionNames) {
      id<MTLFunction> function = [library newFunctionWithName:name];
      if (function.functionType == MTLFunctionTypeKernel &&
          (!kernel_name || [name isEqualToString:kernel_name])) {
        [kernels addObject:function];
      }
    }
    if (!kernels.count && kernel_name) {
      NSMutableArray* names = [NSMutableArray new];
      for (NSString* name in library.functionNames) {
        if ([library newFunctionWithName:name].functionType == MTLFunctionTypeKernel) {
          [names addObject:name];
        }
      }
      fprintf(
          stderr,
          "no kernel %s in %s; its kernels: %s\n",
          kernel_name.UTF8String,
          path.UTF8String,
          names.count ? [names componentsJoinedByString:@" "].UTF8String : "none");
      return 1;
    } else if (!kernels.count) {
      fprintf(stderr, "%s has no kernels\n", path.UTF8String);
      return 1;
    }
    int failed_pipelines = 0;
    printf("%-50s %8s %6s %10s\n", "kernel", "maxTPTG", "simdW", "staticTGmem");
    for (id<MTLFunction> function in kernels) {
      BOOL needs_constants = NO;
      for (MTLFunctionConstant* constant in function.functionConstantsDictionary.allValues) {
        needs_constants = needs_constants || constant.required;
      }
      if (needs_constants) {
        printf(
            "%-50s needs function constants, which pso_info does not set\n",
            function.name.UTF8String);
        failed_pipelines++;
        continue;
      }
      MTLComputePipelineDescriptor* descriptor = [MTLComputePipelineDescriptor new];
      descriptor.computeFunction = function;
      id<MTLComputePipelineState> pipeline =
          [device newComputePipelineStateWithDescriptor:descriptor
                                                options:MTLPipelineOptionNone
                                             reflection:nil
                                                  error:&error];
      if (!pipeline) {
        printf(
            "%-50s pipeline failed: %s\n",
            function.name.UTF8String,
            error.localizedDescription.UTF8String);
        failed_pipelines++;
        continue;
      }
      printf(
          "%-50s %8lu %6lu %10lu\n",
          function.name.UTF8String,
          (unsigned long)pipeline.maxTotalThreadsPerThreadgroup,
          (unsigned long)pipeline.threadExecutionWidth,
          (unsigned long)pipeline.staticThreadgroupMemoryLength);
    }
    return failed_pipelines ? 1 : 0;
  }
}
