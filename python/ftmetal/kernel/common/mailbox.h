// The mailbox protocol between the lookup kernels (moe/moe_lookup, embedding/ple_lookup) and
// their CPU services (service/expert, service/ple): the word layouts, states, errors and sizes,
// and the kernels' parameter blocks. One definition, included by C++ and Metal alike.
//
// A request: the GPU writes its words, increments Sequence and sets State POSTED; the CPU claims it
// (CLAIMED), answers DONE or FAILED, writes the reply and last Acknowledged = Sequence. The
// watchdog answers a stuck request FAILED the same way; a shutdown sets the shutdown word, then
// Acknowledged = Sequence, then State SHUTDOWN. moe_lookup and ple_wait wait in one threadgroup and
// return State to IDLE (SHUTDOWN after a shutdown) themselves. A wait never outlives its CPU: after
// the spin limit unanswered (some 20 s, past the watchdog's timeout, which answers first while the
// CPU lives) it sets the abandoned word and takes the request as unanswered; from then on no kernel
// posts or waits, and check_healthy raises (common/mailbox_wait.h). A process stopped that long (a
// debugger, SIGTSTP) while a wait is pending also fails its services.
#pragma once

#ifdef __METAL_VERSION__
#include <metal_stdlib>
// Each library uses some of the constants: the two lookups and the services share one header.
#define MAILBOX_CONSTANT [[maybe_unused]] constant constexpr
#else
#include <cstdint>
#define MAILBOX_CONSTANT constexpr
namespace ftmetal {
#endif

// The MoE mailbox: uint32 words in 128-byte lines; line 0 the handshake and the shutdown word,
// line 1 the request header, lines 2-9 the request's distinct expert ids (uint16, the absent
// first), line 10 the reply.
enum class MailboxWord : int {
  State = 0,
  // The last request's sequence number, incremented by the GPU per request.
  Sequence = 1,
  // The sequence number of the last request answered.
  Acknowledged = 2,
  // Nonzero once the service shut down; written by the CPU only, never cleared. The GPU
  // posts nothing once it is set, and stops waiting when it appears.
  Shutdown = 3,
  // Nonzero once a GPU wait gave up on the CPU; written by the GPU only, never cleared. Like
  // the shutdown word, it ends every post and wait.
  Abandoned = 4,
  // The spins a GPU wait makes before it gives up, written by the service as it starts.
  SpinLimit = 5,
  Kind = 32,
  Layer = 33,
  Step = 34,
  NumRows = 35,
  NumIds = 36,
  NumMisses = 37,
  Ids = 64,
  NumLoaded = 320,
  NumEvicted = 321,
  Error = 322,
  // uint64 mach_absolute_time stamps.
  ClaimTime = 324,
  DoneTime = 326,
};

MAILBOX_CONSTANT int MAILBOX_WORDS = 1024;
// The id lines hold this many uint16 ids, one threadgroup thread per expert.
MAILBOX_CONSTANT int MAX_LOOKUP_EXPERTS = 512;

enum class MailboxState : uint32_t {
  Idle = 0,
  Posted = 1,
  Claimed = 2,
  Done = 3,
  Failed = 4,
  Shutdown = 5,
};

// Decode and short prefills alike read their misses into the LRU; a probe posts no ids and
// is answered without touching the cache.
enum class MoeLookupKind : uint32_t {
  Decode = 1,
  PrefillCopyMiss = 2,
  Probe = 3,
};

enum class MailboxError : uint32_t {
  None = 0,
  // A read from the checkpoint failed.
  Io = 1,
  // The watchdog found the request unanswered.
  Timeout = 2,
  // The request broke the protocol (sequence, counts, ids) or the host ordering.
  Protocol = 3,
  Shutdown = 4,
  // The layer's pool cannot hold the request's experts.
  Capacity = 5,
};

// The PLE mailbox: uint32 words in 128-byte lines; line 0 the handshake and the shutdown word
// (as MailboxWord's), line 1 the post's header and the reply, lines 2-7 the rows. States
// and errors are MailboxState's and MailboxError's.
enum class PleMailboxWord : int {
  State = 0,
  // The last post's sequence number, incremented by the GPU per post.
  Sequence = 1,
  // The sequence number of the last post answered.
  Acknowledged = 2,
  // Nonzero once the service shut down; written by the CPU only, never cleared.
  Shutdown = 3,
  Abandoned = 4,
  SpinLimit = 5,
  NumRows = 32,
  Error = 33,
  // Per batch row: token, previous token, the one before it.
  Rows = 64,
};

// One wait (common/mailbox_wait.h) serves both mailboxes.
static_assert(
    int(PleMailboxWord::State) == int(MailboxWord::State) &&
        int(PleMailboxWord::Acknowledged) == int(MailboxWord::Acknowledged) &&
        int(PleMailboxWord::Shutdown) == int(MailboxWord::Shutdown) &&
        int(PleMailboxWord::Abandoned) == int(MailboxWord::Abandoned) &&
        int(PleMailboxWord::SpinLimit) == int(MailboxWord::SpinLimit),
    "the two mailboxes share their handshake words");

MAILBOX_CONSTANT int PLE_MAILBOX_WORDS = 256;
// The batch rows one post holds: the row lines' capacity.
MAILBOX_CONSTANT int MAX_PLE_ROWS = 64;

struct LookupParams {
  int layer_id;
  int num_experts;
  int num_rows;
  int top_k;
  uint32_t kind;
  // The outputs are slots less this: the first slot of the layer's pool.
  int slot_base;
};

struct PostParams {
  int num_rows;
};

struct WaitParams {
  // Each part's bytes: num_rows * heads * row bytes.
  int part_bytes[3];
};

#ifndef __METAL_VERSION__
// The SpinLimit word the services write: some 20 s of spinning on an idle GPU, longer on a busy
// one.
constexpr uint32_t MAILBOX_SPIN_LIMIT = 60'000'000;

} // namespace ftmetal
#endif
#undef MAILBOX_CONSTANT
