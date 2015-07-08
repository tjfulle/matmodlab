from sympy import *
NCI = (1,0,2,1,0,3,2,1,0)
NCJ = (0,1,0,1,2,0,1,2,3)

u = ['u = 0.']
for k in range(9):
    i = NCI[k]
    j = NCJ[k]
    w = 'xp({0}) * ((BI1 - 3.) ** {1}) * ((BI2 - 3.) ** {2})'.format(k+1, i, j)
    u.append('u(2) = u(2) + {0}'.format(w))
for k in range(3):
    K = k + 1
    w = '1. / xp({0}) * (Jac - 1.) ** {1}'.format(9+K, 2*K)
    u.append('if (xp({0}) > 0.) u(1) = u(1) + {1}'.format(9+K, w))
u.append('u(1) = u(1) + u(2)')

du = ['DU = 0.']
for k in range(9):
    i = NCI[k]
    j = NCJ[k]
    if i > 0:
        fac = i
        w1 = '(BI1 - 3.) ** {0}'.format(i-1)
        w2 = '(BI2 - 3.) ** {0}'.format(j)
        w = '{0:.1f} * xp({1}) * ({2}) * ({3})'.format(fac, k+1, w1, w2)
        du.append('DU(1) = DU(1) + {0}'.format(w))

    if j > 0:
        fac = j
        w1 = '(BI1 - 3.) ** {0}'.format(i)
        w2 = '(BI2 - 3.) ** {0}'.format(j-1)
        w = '{0:.1f} * xp({1}) * ({2}) * ({3})'.format(fac, k+1, w1, w2)
        du.append('DU(2) = DU(2) + {0}'.format(w))

for k in range(3):
    K = k + 1
    fac = 2 * K
    if fac <= 0:
        continue
    w = '{0:.1f} / xp({1}) * (Jac - 1.) ** {2}'.format(fac, 9+K, 2*K-1)
    du.append('if (xp({0}) > 0.) DU(3) = DU(3) + {1}'.format(9+K, w))

d2u = ['D2U = 0.']
for k in range(9):
    i = NCI[k]
    j = NCJ[k]

    fac = i * (i - 1)
    if fac > 0:
        w1 = '(BI1 - 3.) ** {0}'.format(i-2)
        w2 = '(BI2 - 3.) ** {0}'.format(j)
        w = '{0:.1f} * xp({1}) * ({2}) * ({3})'.format(fac, k+1, w1, w2)
        d2u.append('D2U(1) = D2U(1) + {0}'.format(w))

    fac = j * (j - 1)
    if fac > 0:
        w1 = '(BI1 - 3.) ** {0}'.format(i)
        w2 = '(BI2 - 3.) ** {0}'.format(j-2)
        w = '{0:.1f} * xp({1}) * ({2}) * ({3})'.format(fac, k+1, w1, w2)
        d2u.append('D2U(2) = D2U(2) + {0}'.format(w))

    fac = i * j
    if fac > 0:
        w1 = '(BI1 - 3.) ** {0}'.format(i-1)
        w2 = '(BI2 - 3.) ** {0}'.format(j-1)
        w = '{0:.1f} * xp({1}) * ({2}) * ({3})'.format(fac, k+1, w1, w2)
        d2u.append('D2U(4) = D2U(4) + {0}'.format(w))

for k in range(3):
    K = k + 1
    fac = 2 * K * (2 * K - 1)
    if fac <= 0:
        continue
    w = '{0:.1f} / xp({1}) * (Jac - 1.) ** {2}'.format(fac, 9+K, 2*K-2)
    d2u.append('if (xp({0}) > 0.) D2U(3) = D2U(3) + {1}'.format(9+K, w))

d3u = ['D3U = 0.']
for k in range(3):
    K = k + 1
    fac = 4 * K * (2 * K - 1) * (K - 1)
    if fac <= 0:
        continue
    w = '{0:.1f} / xp({1}) * (Jac - 1.) ** {2}'.format(fac, 9+K, 2*K-3)
    d3u.append('if (xp({0}) > 0.) D3U(6) = D3U(6) + {1}'.format(9+K, w))

with open('uhyper_poly.f90', 'w') as fh:
    fh.write('''\
subroutine uhyper(BI1, BI2, Jac, U, DU, D2U, D3U, temp, noel, cmname, &
     incmpflag, nstatev, statev, nfieldv, fieldv, fieldvinc, &
     nprop, props)
  ! ----------------------------------------------------------------------- !
  ! HYPERELASTIC POLYNOMIAL MODEL
  ! ----------------------------------------------------------------------- !
  implicit none
  character*8, intent(in) :: cmname
  integer, parameter :: dp=selected_real_kind(14)
  integer, intent(in) :: nprop, noel, nstatev, incmpflag, nfieldv
  real(kind=dp), intent(in) :: BI1, BI2, Jac, props(nprop), temp
  real(kind=dp), intent(inout) :: U(2), DU(3), D2U(6), D3U(6), statev(nstatev)
  real(kind=dp), intent(inout) :: fieldv(nfieldv), fieldvinc(nfieldv)
  real(kind=dp) :: xp(nprop)
  ! ----------------------------------------------------------------------- !
  xp = props
''')
    s = 'Energies'
    fh.write('\n  ! {0} {1}\n'.format('-'*(72 - len(s)), s))
    fh.write('  {0}'.format('\n  '.join(u)))

    s = 'First Derivatives'
    fh.write('\n\n  ! {0} {1}\n'.format('-'*(72 - len(s)), s))
    fh.write('  {0}'.format('\n  '.join(du)))

    s = 'Second Derivatives'
    fh.write('\n\n  ! {0} {1}\n'.format('-'*(72 - len(s)), s))
    fh.write('  {0}'.format('\n  '.join(d2u)))

    s = 'Third Derivatives'
    fh.write('\n\n  ! {0} {1}\n'.format('-'*(72 - len(s)), s))
    fh.write('  {0}'.format('\n  '.join(d3u)))

    fh.write('\n\n  return\nend subroutine uhyper')
