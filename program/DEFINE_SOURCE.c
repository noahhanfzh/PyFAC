#include "udf.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <math.h>

#define N 100
#define k0 2.*M_PI/3.3
#define kx M_PI/54.*1000.
#define L 0.1
#define ke 0.74685/L
#define Tu 1.

#define SCALE 500.

DEFINE_SOURCE(x_mom_source, c, d, dS, eqn)
{
srand(1);

real xc[ND_ND];
real x, y, z;
real u, v, w;
real nu, phi;
real kn;

real r1, r2, r3, r4;

real e1x, e1y, e1z;
real e2x, e2y, e2z;
real e3x, e3y, e3z;

real knx, kny, knz;

real alpha;

real thetanx, thetany, thetanz;

real Ek, dk;

real wave_vn;

real vnx, vny, vnz;

real t = CURRENT_TIME;

real prim_u = 0.;
real prim_v = 0.;
real prim_w = 0.;

real source;

C_CENTROID(xc,c,d);
x = xc[0];
y = xc[1];
z = 0;

if (x>-0.5 && x<-0.05 && y>-1.0 && y<1.0)
{
u = C_U(c,d);
v = C_V(c,d);
w = 0.;

for (int n = 0; n < N; n++)
{

r1 = rand() % 1000;
r2 = rand() % 1000;
r3 = rand() % 1000;
r4 = rand() % 1000;

r1 = r1 / 1000.;
r2 = r2 / 1000.;
r3 = r3 / 1000.;
r4 = r4 / 1000.;

kn = k0 * pow(kx / k0, (n + 1. - 0.5) / (double)N);

nu = acos(2. * r1 - 1.);
phi = 2. * M_PI * r2;

e1x = cos(nu) * cos(phi);
e1y = cos(nu) * sin(phi);
e1z = - sin(nu);

e2x = - sin(phi);
e2y = cos(phi);
e2z = 0;

e3x = sin(nu) * cos(phi);
e3y = sin(nu) * sin(phi);
e3z = cos(nu);

knx = kn * e3x;
kny = kn * e3y;
knz = kn * e3z;

alpha = 2. * M_PI * r3;

thetanx = cos(alpha) * e1x + sin(alpha) * e2x;
thetany = cos(alpha) * e1y + sin(alpha) * e2y;
thetanz = cos(alpha) * e1z + sin(alpha) * e2z;

Ek = alpha * pow(Tu * sqrt(u * u + v * v + w * w), 2.) / ke * pow(L, 4.) * pow(kn, 4.) / pow(1 + pow(L, 2.) * pow(kn, 2.), 17./6.);

dk = k0 * (pow(kx / k0, (n + 1.) / (double)N) - pow(kx / k0, (double)n / (double)N));

wave_vn = sqrt(Ek * dk);

vnx = wave_vn * (knx * u + kny * v + knz * w) * sin(knx * (x - u * t) + kny * (y - v * t) + knz * (z - w * t)) * thetanx;
vny = wave_vn * (knx * u + kny * v + knz * w) * sin(knx * (x - u * t) + kny * (y - v * t) + knz * (z - w * t)) * thetany;
vnz = wave_vn * (knx * u + kny * v + knz * w) * sin(knx * (x - u * t) + kny * (y - v * t) + knz * (z - w * t)) * thetanz;

prim_u = prim_u + vnx;
prim_v = prim_v + vny;
prim_w = prim_w + vnz;
}

source = 2. * prim_u / SCALE;
}

else
{
source = 0;
}

return source;

}

DEFINE_SOURCE(y_mom_source, c, d, dS, eqn)
{
srand(1);

real xc[ND_ND];
real x, y, z;
real u, v, w;
real nu, phi;
real kn;

real r1, r2, r3, r4;

real e1x, e1y, e1z;
real e2x, e2y, e2z;
real e3x, e3y, e3z;

real knx, kny, knz;

real alpha;

real thetanx, thetany, thetanz;

real Ek, dk;

real wave_vn;

real vnx, vny, vnz;

real t = CURRENT_TIME;

real prim_u = 0.;
real prim_v = 0.;
real prim_w = 0.;

real source;

C_CENTROID(xc,c,d);
x = xc[0];
y = xc[1];
z = 0;

if (x>-0.5 && x<-0.05 && y>-1.0 && y<1.0)
{
u = C_U(c,d);
v = C_V(c,d);
w = 0.;

for (int n = 0; n < N; n++)
{

r1 = rand() % 1000;
r2 = rand() % 1000;
r3 = rand() % 1000;
r4 = rand() % 1000;

r1 = r1 / 1000.;
r2 = r2 / 1000.;
r3 = r3 / 1000.;
r4 = r4 / 1000.;

kn = k0 * pow(kx / k0, (n + 1. - 0.5) / (double)N);

nu = acos(2. * r1 - 1.);
phi = 2. * M_PI * r2;

e1x = cos(nu) * cos(phi);
e1y = cos(nu) * sin(phi);
e1z = - sin(nu);

e2x = - sin(phi);
e2y = cos(phi);
e2z = 0;

e3x = sin(nu) * cos(phi);
e3y = sin(nu) * sin(phi);
e3z = cos(nu);

knx = kn * e3x;
kny = kn * e3y;
knz = kn * e3z;

alpha = 2. * M_PI * r3;

thetanx = cos(alpha) * e1x + sin(alpha) * e2x;
thetany = cos(alpha) * e1y + sin(alpha) * e2y;
thetanz = cos(alpha) * e1z + sin(alpha) * e2z;

Ek = alpha * pow(Tu * sqrt(u * u + v * v + w * w), 2.) / ke * pow(L, 4.) * pow(kn, 4.) / pow(1 + pow(L, 2.) * pow(kn, 2.), 17./6.);

dk = k0 * (pow(kx / k0, (n + 1.) / (double)N) - pow(kx / k0, (double)n / (double)N));

wave_vn = sqrt(Ek * dk);

vnx = wave_vn * (knx * u + kny * v + knz * w) * sin(knx * (x - u * t) + kny * (y - v * t) + knz * (z - w * t)) * thetanx;
vny = wave_vn * (knx * u + kny * v + knz * w) * sin(knx * (x - u * t) + kny * (y - v * t) + knz * (z - w * t)) * thetany;
vnz = wave_vn * (knx * u + kny * v + knz * w) * sin(knx * (x - u * t) + kny * (y - v * t) + knz * (z - w * t)) * thetanz;

prim_u = prim_u + vnx;
prim_v = prim_v + vny;
prim_w = prim_w + vnz;
}

source = 2. * prim_v / SCALE;
}

else
{
source = 0;
}

return source;

}


DEFINE_SOURCE(z_mom_source, c, d, dS, eqn)
{
srand(1);

real xc[ND_ND];
real x, y, z;
real u, v, w;
real nu, phi;
real kn;

real r1, r2, r3, r4;

real e1x, e1y, e1z;
real e2x, e2y, e2z;
real e3x, e3y, e3z;

real knx, kny, knz;

real alpha;

real thetanx, thetany, thetanz;

real Ek, dk;

real wave_vn;

real vnx, vny, vnz;

real t = CURRENT_TIME;

real prim_u = 0.;
real prim_v = 0.;
real prim_w = 0.;

real source;

C_CENTROID(xc,c,d);
x = xc[0];
y = xc[1];
z = 0;

if (x>-0.5 && x<-0.05 && y>-1.0 && y<1.0)
{
u = C_U(c,d);
v = C_V(c,d);
w = 0.;

for (int n = 0; n < N; n++)
{

r1 = rand() % 1000;
r2 = rand() % 1000;
r3 = rand() % 1000;
r4 = rand() % 1000;

r1 = r1 / 1000.;
r2 = r2 / 1000.;
r3 = r3 / 1000.;
r4 = r4 / 1000.;

kn = k0 * pow(kx / k0, (n + 1. - 0.5) / (double)N);

nu = acos(2. * r1 - 1.);
phi = 2. * M_PI * r2;

e1x = cos(nu) * cos(phi);
e1y = cos(nu) * sin(phi);
e1z = - sin(nu);

e2x = - sin(phi);
e2y = cos(phi);
e2z = 0;

e3x = sin(nu) * cos(phi);
e3y = sin(nu) * sin(phi);
e3z = cos(nu);

knx = kn * e3x;
kny = kn * e3y;
knz = kn * e3z;

alpha = 2. * M_PI * r3;

thetanx = cos(alpha) * e1x + sin(alpha) * e2x;
thetany = cos(alpha) * e1y + sin(alpha) * e2y;
thetanz = cos(alpha) * e1z + sin(alpha) * e2z;

Ek = alpha * pow(Tu * sqrt(u * u + v * v + w * w), 2.) / ke * pow(L, 4.) * pow(kn, 4.) / pow(1 + pow(L, 2.) * pow(kn, 2.), 17./6.);

dk = k0 * (pow(kx / k0, (n + 1.) / (double)N) - pow(kx / k0, (double)n / (double)N));

wave_vn = sqrt(Ek * dk);

vnx = wave_vn * (knx * u + kny * v + knz * w) * sin(knx * (x - u * t) + kny * (y - v * t) + knz * (z - w * t)) * thetanx;
vny = wave_vn * (knx * u + kny * v + knz * w) * sin(knx * (x - u * t) + kny * (y - v * t) + knz * (z - w * t)) * thetany;
vnz = wave_vn * (knx * u + kny * v + knz * w) * sin(knx * (x - u * t) + kny * (y - v * t) + knz * (z - w * t)) * thetanz;

prim_u = prim_u + vnx;
prim_v = prim_v + vny;
prim_w = prim_w + vnz;
}

source = 2. * prim_w / SCALE;
}

else
{
source = 0;
}

return source;

}