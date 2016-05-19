#ifndef _HEAT_SS_H
#define _HEAT_SS_H
#include "defs.h"
void to_supersite(supersite *, float *);
void from_supersite(float *, supersite *);
void lapl_iter_supersite(supersite *, float , supersite *);
#endif /* _HEAT_SS_H */
