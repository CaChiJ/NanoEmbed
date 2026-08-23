// SentencePiece-flavoured BPE with byte fallback — the Gemma 3 family's
// tokenizer, declared in GGUF as `tokenizer.ggml.model == "llama"`.
//
// The "llama" tag names the vocab flavour, not the algorithm. Two arrays
// decide which one applies:
//
//   * `tokenizer.ggml.scores` present and meaningful -> SentencePiece unigram
//   * `tokenizer.ggml.merges`  present              -> rank-ordered BPE
//
// harrier-oss-v1-270m ships both, but its scores are literally 0,1,2,...,N-1
// (the token index, not a log-probability) and it sets the converter's
// `tokenizer.ggml.is_spm_bpe` flag. It is BPE; the scores are filler. Reading
// it as unigram would produce plausible-looking but wrong token IDs.
//
// Pipeline, matching HuggingFace `GemmaTokenizerFast` exactly:
//
//   1. normalize   every ASCII space becomes U+2581 ("▁")
//   2. split       one symbol per UTF-8 character; a character absent from
//                  the vocab becomes one `<0xNN>` token per UTF-8 byte
//   3. merge       repeatedly apply the lowest-rank adjacent pair
//   4. wrap        prepend BOS, append EOS
//
// There is no pre-tokenizer step: the file's Split-on-space rule runs after
// normalization has already removed every space, so BPE sees the whole string
// and merges freely across word boundaries. That is what makes "hello world"
// come out as ["hello", "▁world"] rather than two independently-merged words.
//
// Every symbol at every stage is a vocab entry — initial characters resolve to
// vocab IDs or byte-fallback IDs, and every merge produces a vocab entry — so
// the merge table is keyed by a pair of IDs and encode() builds no strings.
// The full string->ID map is only needed to resolve merges at load time and is
// dropped afterwards; keeping it would cost tens of MB of resident memory for
// a 262k vocab, which matters for a library whose whole point is fitting in
// tens of MB.
//
// Known gap: HuggingFace splits the input on added tokens before normalizing,
// so a literal "<eos>" in the input text becomes the special token there and
// ordinary merged pieces here. Natural text is unaffected, and the parity
// fixture covers the corpus we care about.

#pragma once

#include "tokenizer.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

struct gguf_context;

namespace nanoembed {

class SpmBpeTokenizer : public Tokenizer {
public:
    // Build from GGUF metadata: vocab, merges, and the special-token IDs.
    // Throws TokenizerError when a required array is missing or when the file
    // offers no way to represent an out-of-vocabulary character.
    static SpmBpeTokenizer from_gguf(gguf_context * ctx);

    // Token IDs including the leading BOS and trailing EOS, truncated to fit
    // the effective limit. max_seq_len_override > 0 overrides the configured
    // limit for this call; 0 means "use the configured value".
    std::vector<int> encode(const std::string & text,
                            int                 max_seq_len_override = 0) const override;

    int  max_seq_len() const noexcept override { return max_seq_len_; }
    int  vocab_size()  const noexcept override { return static_cast<int>(vocab_.size()); }
    void set_max_seq_len(int n) override { max_seq_len_ = n; }

    int bos_id() const noexcept { return bos_id_; }
    int eos_id() const noexcept { return eos_id_; }
    int pad_id() const noexcept { return pad_id_; }
    int unk_id() const noexcept { return unk_id_; }

    // What encode() actually wraps the content with, after reconciling the
    // file's flags. -1 means "nothing added on that side".
    int prefix_id() const noexcept { return prefix_id_; }
    int suffix_id() const noexcept { return suffix_id_; }

    const std::vector<std::string> & vocab() const noexcept { return vocab_; }

private:
    struct MergeRule {
        int rank   = 0;   // position in tokenizer.ggml.merges; lower wins
        int merged = -1;  // vocab ID of the concatenation
    };

    std::vector<std::string> vocab_;      // id -> piece

    // Only the single-codepoint pieces, which is all encode() needs to seed
    // the symbol list.
    std::unordered_map<std::string, int> char_ix_;

    // (left_id << 32 | right_id) -> rule.
    std::unordered_map<uint64_t, MergeRule> merges_;

    int byte_id_[256] = {};               // "<0xNN>" ids, -1 when absent
    int bos_id_    = -1;
    int eos_id_    = -1;
    int pad_id_    = -1;
    int unk_id_    = -1;
    int prefix_id_ = -1;
    int suffix_id_ = -1;
    int max_seq_len_ = 512;

    // Append the BPE result for `normalized` to `out`, stopping once `budget`
    // tokens have been produced.
    void bpe(const std::string & normalized, int budget, std::vector<int> & out) const;
};

} // namespace nanoembed
