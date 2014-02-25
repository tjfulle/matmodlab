import numpy

def epsilon(a):
    """Find the machine precision for a float of type 'a'"""
    return numpy.finfo(float).eps


def det9(a):
    """ Determinant of 3x3 array stored as 9x1,
    row major ordering assumed """
    return numpy.linalg.det(numpy.reshape(a, (3, 3)))


def det6(a):
    """ Determinant of 3x3 array stored as 6x1"""
    return numpy.linalg.det(numpy.array([[a[0], a[1], a[2]],
                                         [a[1], a[3], a[4]],
                                         [a[2], a[4], a[5]]]))


def dot(a, b):
    """perform matrix multiplication on two 3x3 matricies"""
    return numpy.dot(a, b)


def u2e(u, kappa):
    """Convert the 3x3 stretch tensor to a strain tensor using the
    Seth-Hill parameter kappa and return a 6x1 array"""
    u2e = numpy.zeros((3, 3))
    if kappa != 0:
        x = 1.0 / kappa * (powm(u, kappa) - numpy.eye(3, 3))
    else:
        x = logm(u)
    return symarray(x)


def symarray(a):
    """Convert a 3x3 matrix to a 6x1 array representing a symmetric matix."""
    mat = (a + a.transpose()) / 2.0
    return numpy.array([mat[0, 0], mat[1, 1], mat[2, 2],
                        mat[0, 1], mat[1, 2], mat[0, 2]])


def asarray(a, n):
    """Convert a 3x3 matrix to array form"""
    if n == 6:
        return symarray(a)
    elif n == 9:
        return numpy.reshape(a, (1, 9))[0]
    else:
        raise Exception("Invalid value for n. Given {0}".format(n))


def as3x3(a):
    """Convert a 6x1 array to a 3x3 symmetric matrix"""
    return numpy.array([[a[0], a[3], a[5]],
                        [a[3], a[1], a[4]],
                        [a[5], a[4], a[2]]])


def expm(a):
    """Compute the matrix exponential of a 3x3 matrix"""
    return funcm(a, numpy.exp)


def powm(a, m):
    """Compute the matrix power of a 3x3 matrix"""
    return funcm(a, lambda x: x ** m)


def sqrtm(a):
    """Compute the square root of a 3x3 matrix"""
    return funcm(a, numpy.sqrt)


def logm(a):
    """Compute the matrix logarithm of a 3x3 matrix"""
    return funcm(a, numpy.log)


def diag(a):
    """Returns the diagonal part of a 3x3 matrix."""
    return numpy.array([[a[0, 0],     0.0,     0.0],
                        [0.0,     a[1, 1],     0.0],
                        [0.0,         0.0, a[2, 2]]])

def isdiag(a):
    """Determines if a matrix is diagonal."""
    return numpy.sum(numpy.abs(a - diag(a))) <= epsilon(a)


def funcm(a, f):
    """Apply function to eigenvalues of a 3x3 matrix then recontruct the matrix
    with the new eigenvalues and the eigenprojectors"""
    if isdiag(a):
        return numpy.array([[f(a[0, 0]),        0.0,        0.0],
                            [       0.0, f(a[1, 1]),        0.0],
                            [       0.0,        0.0, f(a[2, 2])]])

    vals, vecs = numpy.linalg.eig(a)

    # Compute eigenprojections
    p0 = numpy.outer(vecs[:, 0], vecs[:, 0])
    p1 = numpy.outer(vecs[:, 1], vecs[:, 1])
    p2 = numpy.outer(vecs[:, 2], vecs[:, 2])

    return f(vals[0]) * p0 + f(vals[1]) * p1 + f(vals[2]) * p2


def deps2d(dt, k, e, de):
    """
    ! ----------------------------------------------------------------------- !
    ! Compute symmetric part of velocity gradient given depsdt
    ! ----------------------------------------------------------------------- !
    ! Velocity gradient L is given by
    !             L = dFdt * Finv
    !               = dRdt*I*Rinv + R*dUdt*Uinv*Rinv
    ! where F, I, R, U are the deformation gradient, identity, rotation, and
    ! right stretch tensor, respectively. d*dt and *inv are the rate and
    ! inverse or *, respectively,

    ! The stretch U is given by
    !              if k != 0:
    !                  U = (k*E + I)**(1/k)
    !              else:
    !                  U = exp(E)
    ! and its rate
    !                  dUdt = 1/k*(k*E + I)**(1/k - 1)*k*dEdt
    !                       = (k*E + I)**(1/k)*(k*E + I)**(-1)*dEdt
    !                       = U*X*dEdt
    !                  where X = (kE + I)**(-1)
    !    Then
    !              d = sym(L)
    !              w = skew(L)
    """

    D = numpy.zeros((3,3))
    eps = as3x3(e)
    depsdt = as3x3(de)
    epsf = eps + depsdt * dt

    # stretch and its rate
    if k == 0:
        u = expm(epsf)
    else:
        u = powm(k * epsf + numpy.eye(3, 3), 1.0 / k)

    x = 1.0 / 2.0 * (numpy.linalg.inv(k * epsf + numpy.eye(3, 3)) +
                     numpy.linalg.inv(k * eps + numpy.eye(3, 3)))
    du = numpy.dot(numpy.dot(u, x), depsdt)

    L = numpy.dot(du, numpy.linalg.inv(u))
    D = (L + L.transpose()) / 2.0

    return symarray(D)


def update_deformation(dt, k, farg, darg):
    """
    ! ----------------------------------------------------------------------- !
    ! From the value of the Seth-Hill parameter kappa, current strain E,
    ! deformation gradient F, and symmetric part of the velocit gradient d,
    ! update the strain and deformation gradient.
    ! ----------------------------------------------------------------------- !
    ! Deformation gradient is given by
    !
    !              dFdt = L*F                                             (1)
    !
    ! where F and L are the deformation and velocity gradients, respectively.
    ! The solution to (1) is
    !
    !              F = F0*exp(Lt)
    !
    ! Solving incrementally,
    !
    !              Fc = Fp*exp(Lp*dt)
    !
    ! where the c and p refer to the current and previous values, respectively.
    !
    ! With the updated F, Fc, known, the updated stretch is found by
    !
    !              U = (trans(Fc)*Fc)**(1/2)
    !
    ! Then, the updated strain is found by
    !
    !              E = 1/k * (U**k - I)
    !
    ! where k is the Seth-Hill strain parameter.
    """

    f0 = farg.reshape((3, 3))
    d = as3x3(darg)
    ff = numpy.dot(expm(d * dt), f0)
    u = sqrtm(numpy.dot(ff.transpose(), ff))
    if k == 0:
        eps = logm(u)
    else:
        eps = 1.0 / k * (powm(u, k) - numpy.eye(3, 3))

    if numpy.linalg.det(ff) <= 0.0:
        raise Exception("negative jacobian encountered")

    f = asarray(ff, 9)
    e = symarray(eps)

    return f, e



