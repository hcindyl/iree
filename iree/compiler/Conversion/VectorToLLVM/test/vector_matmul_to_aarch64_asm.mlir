module {
    //func @vector_matmul_to_aarch64_asm(%lhs: vector<4x4xi8>, %rhs: vector<4x4xi8>) ->  vector<16xi32> {
    //    %lhs_vec = vector.shape_cast %lhs : vector<4x4xi8> to vector<16xi8>
    //    %rhs_vec = vector.shape_cast %rhs : vector<4x4xi8> to vector<16xi8>
    //    %lhs_vec32 = sexti %lhs_vec : vector<16xi8> to vector<16xi32>
    //    %rhs_vec32 = sexti %rhs_vec : vector<16xi8> to vector<16xi32>
    //    %result_vec = vector.matrix_multiply %lhs_vec32, %rhs_vec32  {lhs_columns = 4 : i32, lhs_rows = 4 : i32, rhs_columns = 4 : i32} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
        //%result = vector.shape_cast %result_vec : vector<16xi32> to vector<4x4xi32>
    //    return %result_vec : vector<16xi32>
    //}


     //func @vector_matmul_to_aarch64_asm(%lhs: memref<4x4xi8>, %rhs: memref<4x4xi8>, %dst: memref<4x4xi32>) {
    func @vector_matmul_to_aarch64_asm(%lhs: vector<4x4xi8>, %rhs: vector<4x4xi8>, %dst: vector<4x4xi32>) -> vector<4x4xi32> {
        %0 = vector.contract {
            indexing_maps = [
                affine_map<(d0, d1, d2) -> (d0, d2)>,
                affine_map<(d0, d1, d2) -> (d2, d1)>,
                affine_map<(d0, d1, d2) -> (d0, d1)>
            ], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>
        } %lhs, %rhs, %dst : vector<4x4xi8>, vector<4x4xi8> into vector<4x4xi32>
        return %0 : vector<4x4xi32>
        //%c0 = constant 0 : index 
        //%c0_i8 = constant 0 : i8
        //%c0_i32 = constant 0 : i32
        //%0 = vector.transfer_read %lhs[%c0, %c0], %c0_i8 {masked = [false, false]} : memref<4x4xi8>, vector<4x4xi8>
        //%1 = vector.transfer_read %rhs[%c0, %c0], %c0_i8 {masked = [false, false]} : memref<4x4xi8>, vector<4x4xi8>
        //%2 = vector.transfer_read %dst[%c0, %c0], %c0_i32 {masked = [false, false]} : memref<4x4xi32>, vector<4x4xi32>
        //%3 = sexti %0 : vector<4x4xi8> to vector<4x4xi32>
        //%4 = sexti %1 : vector<4x4xi8> to vector<4x4xi32>
        //%5 = addi %3, %4 : vector<4x4xi32>
        //vector.transfer_write %5, %dst[%c0, %c0] {masked = [false, false]}: vector<4x4xi32>, memref<4x4xi32>
        //return
    }
}