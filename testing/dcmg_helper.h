// =============================================================================
// integer functions

/// For integers x >= 0, y > 0, returns ceil( x/y ).
/// For x == 0, this is 0.
__host__ __device__
static inline int ceildiv( int x, int y )
{
    return (x + y - 1)/y;
}