#pragma once

inline constexpr char METAL_COOPERATIVE_MATRIX[] = R"msl(
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

template<typename T, uint Rows, uint Columns>
struct evk_cooperative_matrix {
    static_assert(Rows % 8 == 0 && Columns % 8 == 0, "Cooperative dimensions must be multiples of eight");
    using storage_type = T[Rows * Columns / 32];
    simdgroup_matrix<T, 8, 8> tiles[Rows / 8][Columns / 8];

    evk_cooperative_matrix() = default;
    evk_cooperative_matrix(T value) {
        for (uint row = 0; row < Rows / 8; ++row)
            for (uint column = 0; column < Columns / 8; ++column)
                tiles[row][column] = make_filled_simdgroup_matrix<T, 8, 8>(value);
    }
    template<typename U>
    evk_cooperative_matrix(evk_cooperative_matrix<U, Rows, Columns> source) {
        for (uint index = 0; index < Rows * Columns / 32; ++index) set(index, T(source.get(index)));
    }
    T get(uint index) const {
        return tiles[index / (Columns / 8 * 2)][index / 2 % (Columns / 8)].thread_elements()[index % 2];
    }
    void set(uint index, T value) {
        tiles[index / (Columns / 8 * 2)][index / 2 % (Columns / 8)].thread_elements()[index % 2] = value;
    }
    struct Element {
        thread evk_cooperative_matrix* matrix;
        uint index;
        operator T() const { return matrix->get(index); }
        thread Element& operator=(T value) { matrix->set(index, value); return *this; }
    };
    Element operator[](uint index) { return {this, index}; }
    T operator[](uint index) const { return get(index); }
};

template<typename T, uint Rows, uint Columns, typename Pointer>
void simdgroup_load(thread evk_cooperative_matrix<T, Rows, Columns>& matrix, Pointer data,
        ulong stride, ulong2 origin = ulong2(0), bool columnMajor = false) {
    for (uint row = 0; row < Rows / 8; ++row)
        for (uint column = 0; column < Columns / 8; ++column)
            simdgroup_load(matrix.tiles[row][column], data, stride,
                origin + (columnMajor ? ulong2(row * 8, column * 8) : ulong2(column * 8, row * 8)), columnMajor);
}

template<typename T, uint Rows, uint Columns, typename Pointer>
void simdgroup_store(evk_cooperative_matrix<T, Rows, Columns> matrix, Pointer data,
        ulong stride, ulong2 origin = ulong2(0), bool columnMajor = false) {
    for (uint row = 0; row < Rows / 8; ++row)
        for (uint column = 0; column < Columns / 8; ++column)
            simdgroup_store(matrix.tiles[row][column], data, stride,
                origin + (columnMajor ? ulong2(row * 8, column * 8) : ulong2(column * 8, row * 8)), columnMajor);
}

template<typename T, typename U, typename V, uint Rows, uint Columns, uint Inner>
void simdgroup_multiply_accumulate(thread evk_cooperative_matrix<T, Rows, Columns>& result,
        evk_cooperative_matrix<U, Rows, Inner> left, evk_cooperative_matrix<V, Inner, Columns> right,
        evk_cooperative_matrix<T, Rows, Columns> previous) {
    result = previous;
    for (uint row = 0; row < Rows / 8; ++row)
        for (uint column = 0; column < Columns / 8; ++column)
            for (uint inner = 0; inner < Inner / 8; ++inner)
                simdgroup_multiply_accumulate(result.tiles[row][column], left.tiles[row][inner],
                    right.tiles[inner][column], result.tiles[row][column]);
}

template<typename T, uint Rows, uint Columns>
evk_cooperative_matrix<T, Rows, Columns> operator+(evk_cooperative_matrix<T, Rows, Columns> left,
        evk_cooperative_matrix<T, Rows, Columns> right) {
    for (uint index = 0; index < Rows * Columns / 32; ++index) left.set(index, left.get(index) + right.get(index));
    return left;
}

template<typename T, uint Rows, uint Columns>
evk_cooperative_matrix<T, Rows, Columns> operator*(evk_cooperative_matrix<T, Rows, Columns> left,
        evk_cooperative_matrix<T, Rows, Columns> right) {
    for (uint index = 0; index < Rows * Columns / 32; ++index) left.set(index, left.get(index) * right.get(index));
    return left;
}

template<typename T, uint Rows, uint Columns>
evk_cooperative_matrix<T, Rows, Columns> operator*(evk_cooperative_matrix<T, Rows, Columns> matrix, T value) {
    for (uint index = 0; index < Rows * Columns / 32; ++index) matrix.set(index, matrix.get(index) * value);
    return matrix;
}
)msl";
