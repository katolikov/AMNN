//
//  BinCountTest.cpp
//  MNNTests
//

#include <vector>
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include "MNNTestSuite.h"
#include "TestUtils.h"

using namespace MNN::Express;

class BinCountTest : public MNNTestCase {
public:
    virtual ~BinCountTest() = default;

    virtual bool run(int precision) {
        // ===== case 1: small unweighted, hand-computed =====
        {
            VARP input = _Input({10}, NCHW, halide_type_of<int>());
            std::vector<int> data = {0, 1, 1, 2, 2, 2, 5, 5, 15, 15};
            ::memcpy(input->writeMap<int>(), data.data(), data.size() * sizeof(int));

            auto output = _BinCount(input, 16);
            auto got = output->readMap<int>();
            std::vector<int> expected = {1, 2, 3, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2};
            for (int i = 0; i < 16; ++i) {
                if (got[i] != expected[i]) {
                    MNN_ERROR("BinCountTest case1 (small unweighted) failed at bin %d: got %d, want %d\n",
                              i, got[i], expected[i]);
                    return false;
                }
            }
        }

        // ===== case 2: out-of-range values (<0 or >=binNum) are dropped =====
        {
            VARP input = _Input({6}, NCHW, halide_type_of<int>());
            std::vector<int> data = {-1, 0, 3, 4, 4, 100};
            ::memcpy(input->writeMap<int>(), data.data(), data.size() * sizeof(int));

            auto output = _BinCount(input, 4);
            auto got = output->readMap<int>();
            std::vector<int> expected = {1, 0, 0, 1}; // -1,4,4,100 dropped; 0->bin0, 3->bin3
            for (int i = 0; i < 4; ++i) {
                if (got[i] != expected[i]) {
                    MNN_ERROR("BinCountTest case2 (out-of-range drop) failed at bin %d: got %d, want %d\n",
                              i, got[i], expected[i]);
                    return false;
                }
            }
        }

        // ===== case 3: weighted -> float weight-sums =====
        {
            VARP input   = _Input({5}, NCHW, halide_type_of<int>());
            VARP weights = _Input({5}, NCHW, halide_type_of<float>());
            std::vector<int>   idx = {0, 1, 1, 3, 3};
            std::vector<float> w   = {0.5f, 1.0f, 2.0f, 0.25f, 0.75f};
            ::memcpy(input->writeMap<int>(), idx.data(), idx.size() * sizeof(int));
            ::memcpy(weights->writeMap<float>(), w.data(), w.size() * sizeof(float));

            auto output = _BinCount(input, 4, weights);
            auto got = output->readMap<float>();
            std::vector<float> expected = {0.5f, 3.0f, 0.0f, 1.0f};
            if (!checkVectorByRelativeError<float>(got, expected.data(), expected.size(), 0.001f)) {
                MNN_ERROR("BinCountTest case3 (weighted) failed!\n");
                return false;
            }
        }

        // ===== case 4: full [1,1,1440,1920] shape (target on-device use case) =====
        // values = i % 16, so every bin gets exactly n/16 = 172800 counts.
        {
            const int H = 1440, W = 1920;
            const int n = H * W;
            VARP input = _Input({1, 1, H, W}, NCHW, halide_type_of<int>());
            auto ptr = input->writeMap<int>();
            for (int i = 0; i < n; ++i) {
                ptr[i] = i % 16;
            }
            auto output = _BinCount(input, 16);
            auto got = output->readMap<int>();
            const int expectedPerBin = n / 16; // 172800
            for (int i = 0; i < 16; ++i) {
                if (got[i] != expectedPerBin) {
                    MNN_ERROR("BinCountTest case4 (full shape) failed at bin %d: got %d, want %d\n",
                              i, got[i], expectedPerBin);
                    return false;
                }
            }
        }

        // ===== case 5: full shape, fully skewed (all elements -> bin 0) =====
        // Worst case for the naive global-atomic path (every work-item contends
        // on output[0]); the register-histogram path is distribution-independent.
        {
            const int H = 1440, W = 1920;
            const int n = H * W;
            VARP input = _Input({1, 1, H, W}, NCHW, halide_type_of<int>());
            auto ptr = input->writeMap<int>();
            for (int i = 0; i < n; ++i) {
                ptr[i] = 0;
            }
            auto output = _BinCount(input, 16);
            auto got = output->readMap<int>();
            if (got[0] != n) {
                MNN_ERROR("BinCountTest case5 (skewed) failed at bin 0: got %d, want %d\n", got[0], n);
                return false;
            }
            for (int i = 1; i < 16; ++i) {
                if (got[i] != 0) {
                    MNN_ERROR("BinCountTest case5 (skewed) failed at bin %d: got %d, want 0\n", i, got[i]);
                    return false;
                }
            }
        }

        // ===== case 6: float input (integer-valued) =====
        // Exercises the float-input GPU kernel: on OpenCL Low precision the
        // device buffer is fp16 (half), on CPU / high precision it is fp32.
        // Values 0..15 are fp16-exact, so counts are exact either way.
        {
            const int H = 1440, W = 1920;
            const int n = H * W;
            VARP input = _Input({1, 1, H, W}, NCHW, halide_type_of<float>());
            auto ptr = input->writeMap<float>();
            for (int i = 0; i < n; ++i) {
                ptr[i] = (float)(i % 16);
            }
            auto output = _BinCount(input, 16);
            auto got = output->readMap<int>();
            const int expectedPerBin = n / 16;
            for (int i = 0; i < 16; ++i) {
                if (got[i] != expectedPerBin) {
                    MNN_ERROR("BinCountTest case6 (float input) failed at bin %d: got %d, want %d\n",
                              i, got[i], expectedPerBin);
                    return false;
                }
            }
        }

        // ===== case 7: binary mask over the full [1,1,1440,1920] frame =====
        // value[i] = i%16, mask[i] = (i%2==0). Since i%16==b implies i and b
        // share parity, even bins keep all n/16 elements and odd bins keep none
        // -- a deterministic check of the masked-count path (int32 value+mask).
        {
            const int H = 1440, W = 1920;
            const int n = H * W;
            VARP input = _Input({1, 1, H, W}, NCHW, halide_type_of<int>());
            VARP mask  = _Input({1, 1, H, W}, NCHW, halide_type_of<int>());
            auto ip = input->writeMap<int>();
            auto mp = mask->writeMap<int>();
            for (int i = 0; i < n; ++i) {
                ip[i] = i % 16;
                mp[i] = (i % 2 == 0) ? 1 : 0;
            }
            auto output = _BinCount(input, 16, mask, /*binaryMask=*/true);
            auto got = output->readMap<int>();
            const int keptPerEvenBin = n / 16;
            for (int b = 0; b < 16; ++b) {
                const int want = (b % 2 == 0) ? keptPerEvenBin : 0;
                if (got[b] != want) {
                    MNN_ERROR("BinCountTest case7 (int mask) failed at bin %d: got %d, want %d\n",
                              b, got[b], want);
                    return false;
                }
            }
        }

        // ===== case 8: binary mask, float value + float mask (fp16 buffer) =====
        // Same construction as case 7 but exercises BINCOUNT_IN_FLOAT +
        // BINCOUNT_MASK_FLOAT (both stored as half under OpenCL Low precision).
        {
            const int H = 1440, W = 1920;
            const int n = H * W;
            VARP input = _Input({1, 1, H, W}, NCHW, halide_type_of<float>());
            VARP mask  = _Input({1, 1, H, W}, NCHW, halide_type_of<float>());
            auto ip = input->writeMap<float>();
            auto mp = mask->writeMap<float>();
            for (int i = 0; i < n; ++i) {
                ip[i] = (float)(i % 16);
                mp[i] = (i % 2 == 0) ? 1.0f : 0.0f;
            }
            auto output = _BinCount(input, 16, mask, /*binaryMask=*/true);
            auto got = output->readMap<int>();
            const int keptPerEvenBin = n / 16;
            for (int b = 0; b < 16; ++b) {
                const int want = (b % 2 == 0) ? keptPerEvenBin : 0;
                if (got[b] != want) {
                    MNN_ERROR("BinCountTest case8 (float mask) failed at bin %d: got %d, want %d\n",
                              b, got[b], want);
                    return false;
                }
            }
        }

        // ===== case 9: sample_stride=8 downsampling over the full frame =====
        // value[h,w] = h%16. Sampled rows h=0,8,16,.. give h%16 alternating
        // 0,8, so bins 0 and 8 each get (Hs/2)*Ws counts, everything else 0.
        {
            const int H = 1440, W = 1920, S = 8;
            const int n = H * W;
            const int Hs = (H + S - 1) / S, Ws = (W + S - 1) / S;
            VARP input = _Input({1, 1, H, W}, NCHW, halide_type_of<int>());
            auto ip = input->writeMap<int>();
            for (int i = 0; i < n; ++i) {
                ip[i] = (i / W) % 16;
            }
            auto output = _BinCount(input, 16, nullptr, /*binaryMask=*/false, /*sampleStride=*/S);
            auto got = output->readMap<int>();
            const int perHitBin = (Hs / 2) * Ws;   // rows with h%16==0 (and ==8)
            for (int b = 0; b < 16; ++b) {
                const int want = (b == 0 || b == 8) ? perHitBin : 0;
                if (got[b] != want) {
                    MNN_ERROR("BinCountTest case9 (stride=8) failed at bin %d: got %d, want %d\n",
                              b, got[b], want);
                    return false;
                }
            }
        }

        // ===== case 10: sample_stride=8 + binary mask, float value + mask =====
        // value[h,w]=h%16 (bins 0/8); mask keeps sampled columns with ws even
        // (mask = ((w/S)%2==0)), so bins 0 and 8 get (Hs/2)*(Ws/2) counts.
        {
            const int H = 1440, W = 1920, S = 8;
            const int n = H * W;
            const int Hs = (H + S - 1) / S, Ws = (W + S - 1) / S;
            VARP input = _Input({1, 1, H, W}, NCHW, halide_type_of<float>());
            VARP mask  = _Input({1, 1, H, W}, NCHW, halide_type_of<float>());
            auto ip = input->writeMap<float>();
            auto mp = mask->writeMap<float>();
            for (int i = 0; i < n; ++i) {
                ip[i] = (float)((i / W) % 16);
                mp[i] = (((i % W) / S) % 2 == 0) ? 1.0f : 0.0f;
            }
            auto output = _BinCount(input, 16, mask, /*binaryMask=*/true, /*sampleStride=*/S);
            auto got = output->readMap<int>();
            const int perHitBin = (Hs / 2) * (Ws / 2);
            for (int b = 0; b < 16; ++b) {
                const int want = (b == 0 || b == 8) ? perHitBin : 0;
                if (got[b] != want) {
                    MNN_ERROR("BinCountTest case10 (stride+mask) failed at bin %d: got %d, want %d\n",
                              b, got[b], want);
                    return false;
                }
            }
        }

        // ===== case 11: binNum=256 -> local-memory histogram path =====
        // binNum>16 dispatches to bincount_local_buf on GPU. value=i%256 over
        // 512x512 gives every bin exactly 512x512/256 = 1024 counts.
        {
            const int H = 512, W = 512;
            const int n = H * W;
            VARP input = _Input({1, 1, H, W}, NCHW, halide_type_of<int>());
            auto ptr = input->writeMap<int>();
            for (int i = 0; i < n; ++i) {
                ptr[i] = i % 256;
            }
            auto output = _BinCount(input, 256);
            auto got = output->readMap<int>();
            const int expectedPerBin = n / 256; // 1024
            for (int b = 0; b < 256; ++b) {
                if (got[b] != expectedPerBin) {
                    MNN_ERROR("BinCountTest case11 (local 256-bin) failed at bin %d: got %d, want %d\n",
                              b, got[b], expectedPerBin);
                    return false;
                }
            }
        }

        // ===== case 12: local path + binary mask + stride together =====
        // binNum=32 (>16 -> local), stride=2, mask keeps ws-even columns.
        // value[h,w]=h%32; sampled h=hs*2 -> value=2*(hs%16) (even values only),
        // each even value V has 8 sampled rows; mask keeps 64 of 128 columns.
        // So bins {0,2,..,30} get 8*64=512, everything else 0.
        {
            const int H = 256, W = 256, S = 2;
            const int n = H * W;
            VARP input = _Input({1, 1, H, W}, NCHW, halide_type_of<int>());
            VARP mask  = _Input({1, 1, H, W}, NCHW, halide_type_of<int>());
            auto ip = input->writeMap<int>();
            auto mp = mask->writeMap<int>();
            for (int i = 0; i < n; ++i) {
                ip[i] = (i / W) % 32;
                mp[i] = (((i % W) / S) % 2 == 0) ? 1 : 0;
            }
            auto output = _BinCount(input, 32, mask, /*binaryMask=*/true, /*sampleStride=*/S);
            auto got = output->readMap<int>();
            for (int b = 0; b < 32; ++b) {
                const int want = (b % 2 == 0) ? 512 : 0;
                if (got[b] != want) {
                    MNN_ERROR("BinCountTest case12 (local+mask+stride) failed at bin %d: got %d, want %d\n",
                              b, got[b], want);
                    return false;
                }
            }
        }

        return true;
    }
};

MNNTestSuiteRegister(BinCountTest, "op/BinCount");
