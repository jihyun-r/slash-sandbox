#include <nih/basic/types.h>

namespace pbgi {

struct PBGI_state
{
    float*      x;
    float*      y;
    float*      z;
    float*      nx;
    float*      ny;
    float*      nz;
};
struct PBGI_values
{
    float*      r;
    float*      g;
    float*      b;
};

namespace gpu {

void test_pbgi(const nih::uint32 n_points);

} // namespace gpu

} // namespace pbgi