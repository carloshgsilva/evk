// Three numeric FP16 components preserve reduction residuals. The scaled
// high component covers long-context sums; the tail retains tiny gradients.
// Include after declaring BufferFp16. No FP32 value is stored in a buffer.
float read_fp16_stat(BufferFp16 data, uint i) {
    return (float(data.x[i]) * 256.0 + float(data.x[i + 1u])) + float(data.x[i + 2u]) / 65536.0;
}
float truncate_fp16(float x) {
    // Explicit rounding is required: the compiler can elide cast roundtrips.
    if (abs(x) < 0.00006103515625)
        return sign(x) * floor(abs(x) * 16777216.0) / 16777216.0;
    return uintBitsToFloat(floatBitsToUint(x) & 0xffffe000u);
}
void write_fp16_stat(BufferFp16 data, uint i, float x) {
    float high = truncate_fp16(x / 256.0);
    precise float residual = x - high * 256.0;
    float middle = truncate_fp16(residual);
    data.x[i] = float16_t(high);
    data.x[i + 1u] = float16_t(middle);
    precise float tail = (residual - middle) * 65536.0;
    data.x[i + 2u] = float16_t(tail);
}
