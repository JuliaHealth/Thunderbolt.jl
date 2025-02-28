abstract type Smoother end
abstract type Sweep end
struct SymmetricSweep <: Sweep end
struct ForwardSweep   <: Sweep end
struct BackwardSweep  <: Sweep end

struct GaussSeidel{S} <: Smoother
    sweep::S
    iter::Int
end

GaussSeidel(; iter = 1) = GaussSeidel(SymmetricSweep(), iter)
GaussSeidel(f::ForwardSweep) = GaussSeidel(f, 1)
GaussSeidel(b::BackwardSweep) = GaussSeidel(b, 1)
GaussSeidel(s::SymmetricSweep) = GaussSeidel(s, 1)

function (s::GaussSeidel{S})(A, x, b) where {S<:Sweep}
    for i in 1:s.iter
        if S === ForwardSweep || S === SymmetricSweep
            gs!(A, b, x, 1, 1, size(A, 1))
        end
        if S === BackwardSweep || S === SymmetricSweep
            gs!(A, b, x, size(A, 1), -1, 1)
        end
    end
end

function gs!(A, b, x, start, step, stop)
    z = zero(eltype(A))
    @inbounds for col in 1:size(x, 2)
        for i in start:step:stop
            rsum = z
            d = z
            for j in nzrange(A, i)
                row = A.rowval[j]
                val = A.nzval[j]
                d = ifelse(i == row, val, d)
                rsum += ifelse(i == row, z, val * x[row, col])
            end
            x[i, col] = ifelse(d == 0, x[i, col], (b[i, col] - rsum) / d)
        end
    end
end


struct CSCPartition
    owned_rows::UnitRange{Int}   # Owned rows.
    owned_cols::UnitRange{Int}   # Owned columns.
    ghost_rows::Vector{Int}     # this actually the owned rows that have ghost columns, so naming ?!
    ghost_cols::Vector{Int}     # Additional column indices required for the update.
end


struct PartitionedSparseMatrixCSC{Tv,Ti}
    A::SparseMatrixCSC{Tv,Ti}
    partitions::Vector{CSCPartition}
end

function partition_matrix(A::SparseMatrixCSC, nparts::Int)
    N = size(A, 1)
    partitions = CSCPartition[]
    # Compute start and end indices for each partition.
    starts = [floor(Int, (i-1) * N / nparts) + 1 for i in 1:nparts]
    ends   = [floor(Int, i * N / nparts) for i in 1:nparts]

    for i in 1:nparts
        owned_rows = starts[i]:ends[i]
        owned_cols = owned_rows

        # Determine extra indices from neighboring partitions.
        ghost_rows = Int[]
        ghost_cols = Int[]

        # If not the first partition, use the first owned row (in this partition)
        # as the row associated with the ghost column from the previous partition.
        if i > 1
            push!(ghost_rows, first(owned_rows))
            push!(ghost_cols, ends[i-1])
        end
        # If not the last partition, use the last owned row (in this partition)
        # as the row associated with the ghost column from the next partition.
        if i < nparts
            push!(ghost_rows, last(owned_rows))
            push!(ghost_cols, starts[i+1])
        end

        push!(partitions, CSCPartition(owned_rows, owned_cols, ghost_rows, ghost_cols))
    end
    return PartitionedSparseMatrixCSC(A, partitions)
end



function row_iter(A, i::Int)
    out = Tuple{Int, eltype(A)}[]
    for j in 1:size(A, 2)
        for k in A.colptr[j]:(A.colptr[j+1]-1)
            if A.rowval[k] == i
                push!(out, (j, A.nzval[k]))
            end
        end
    end
    return out
end


function partition_gs!(A::SparseMatrixCSC, b, x, part::CSCPartition, sweep::Sweep)
    # TODO: add D_L1
    # Determine the ordering of owned rows based on sweep direction.
    owned = collect(part.owned_rows)
    local_rows = sweep isa BackwardSweep ? reverse(owned) : owned

    # Allowed columns: owned plus extra.
    allowed_cols = union(Set(part.owned_cols), Set(part.ghost_cols))
    for i in local_rows
        rsum = zero(eltype(A))
        d = zero(eltype(A))
        for (j, val) in row_iter(A, i)
            if j in allowed_cols
                if i == j
                    d = val
                else
                    rsum += val * x[j]
                end
            end
        end
        if d != 0
            x[i] = (b[i] - rsum) / d
        end
    end
end



struct L1GaussSeidel{S<:Sweep}
    sweep::S
    iter::Int
end

L1GaussSeidel(; iter = 1) = L1GaussSeidel(SymmetricSweep(), iter)
L1GaussSeidel(f::ForwardSweep) = L1GaussSeidel(f, 1)
L1GaussSeidel(b::BackwardSweep) = L1GaussSeidel(b, 1)
L1GaussSeidel(s::SymmetricSweep) = L1GaussSeidel(s, 1)


function (s::L1GaussSeidel{S})(PS::PartitionedSparseMatrixCSC, x, b) where {S<:Sweep}
    for k in 1:s.iter
        for part in PS.partitions
            if s.sweep isa ForwardSweep || s.sweep isa SymmetricSweep
                partition_gs!(PS.A, b, x, part, ForwardSweep())
            end
            if s.sweep isa BackwardSweep || s.sweep isa SymmetricSweep
                partition_gs!(PS.A, b, x, part, BackwardSweep())
            end
        end
    end
end


## Example usage
## normal Gauss-Seidel
N = 4
A = spdiagm(0 => 2 * ones(N), -1 => -ones(N-1), 1 => -ones(N-1))
x = eltype(A).(collect(0:N-1))
b = zeros(N)
s = GaussSeidel(ForwardSweep())
s(A, x, b)
x




# Create the partitioned sparse matrix structure.
## TODO: maybe add Base.show to print the partitions
pA = partition_matrix(A, 2)
px = eltype(A).(collect(0:N-1))
pb = zeros(N)
l1 = L1GaussSeidel(ForwardSweep())
l1(pA, px, pb)
