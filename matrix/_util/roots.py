import math
import cmath
import scipy.special

_SQRT_THREE = math.sqrt(3)


def solver_depressed_cubic(p, q, include_complex_values=True):
    # compute discriminant
    D = (p / 3)**3 + (q / 2)**2

    # one real root and two complex roots
    if D > 0:
        # compute axiliary values u and v
        D_sqrt = math.sqrt(D)
        u = - 0.5 * q + D_sqrt
        v = - 0.5 * q - D_sqrt
        u = scipy.special.cbrt(float(u))
        v = scipy.special.cbrt(float(v))
        assert math.isclose(u * v, - p / 3, abs_tol=1e-08)

        # compute roots
        x1 = u + v
        if include_complex_values:
            a = - 0.5 + 0.5j * _SQRT_THREE
            b = - 0.5 - 0.5j * _SQRT_THREE
            x2 = u * a + v * b
            x3 = u * b + v * a
            roots = (x1, x2, x3)
        else:
            roots = (x1,)

    # three real roots
    elif D < 0:
        # compute axiliary values u and v
        D_sqrt = cmath.sqrt(D)
        u = - 0.5 * q + D_sqrt
        v = - 0.5 * q - D_sqrt
        u = u**(1 / 3)
        v = v**(1 / 3)
        assert cmath.isclose(u * v, - p / 3, rel_tol=1e-06, abs_tol=1e-06)

        # compute roots
        x1 = u + v
        a = - 0.5 + 0.5j * _SQRT_THREE
        b = - 0.5 - 0.5j * _SQRT_THREE
        x2 = u * a + v * b
        x3 = u * b + v * a
        roots = (x1, x2, x3)
        assert all(math.isclose(x.imag, 0, abs_tol=1e-08) for x in roots)
        roots = [x.real for x in roots]

    # one or two real roots
    else:
        # zero is triple root
        if p == 0:
            assert q == 0
            x1 = 0
            roots = (x1,)
        # single real root and double real root
        else:
            x1 = 3 * q / p
            x2 = - 0.5 * x1
            roots = (x1, x2)

    # return
    assert all(cmath.isclose(x**3 + p * x + q, 0, abs_tol=1e-08) for x in roots)
    return roots
