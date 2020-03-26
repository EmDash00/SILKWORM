import numpy as np


class RangeError(ValueError):
    pass


def clamp(x, limits):
    if (len(limits) != 2):
        raise ValueError(
            "x_range must a list or tuple in the format (min, max)")
    if (limits[0] > limits[1]):
        raise RangeError("limits must be of form (min, max)")

    return max(limits[0], min(x, limits[1]))


def in_range(x, valid_range):
    if (valid_range[0] >= valid_range[1]):
        raise RangeError("valid_range must be in the format (min, max)")
    return (x < valid_range[1] and x > valid_range[0])


def calc_1daffine_map(domain, image, strict=False):

    m = (image[1] - image[0]) / (domain[1] - domain[0])
    b = -m * domain[0] + image[0]

    if (domain[0] > domain[1]):
        raise RangeError("valid_range must be in the format (min, max)")

    if (strict):

        def map(x):
            if (not in_range(x, domain)):
                raise RangeError("x is out of range of the domain")

            return (m * x + b)

        return (map)
    else:
        return (lambda x: (m * x + b))


def calc_ndlinear_map(domain, image):
    """
    Calculates an N-dimensional linear map from R^n -> R^n (ND -> ND)
    given the domain and image of the transformation and returns it
    as a lambda.

    :param domain: The domain of the transformation expressed as a matrix of
    the basis vectors [x, y, z, ...]

    :param image: The image of the transformation expressed as the set of basis
    vectors under the transformation.
    """
    A = np.np.array(domain)
    B = np.np.array(image)
    T = np.linalg.inv(A) * B

    return (lambda x: T * x)


def circular_map(theta, domain):
    """
    Defines a circular domain (like angles between 0 and 2np.pi) to map x to.

    :param x: The scalar to be mapped
    :param domain: A tuple of values expresnp.sing the domain as a range.
    """
    if (domain[1] <= domain[0]):
        raise RangeError("The minimum must be less than the maximum.")

    gamma = domain[1] - domain[0]
    return (theta - gamma * ((theta - domain[0]) // gamma))


class Vector:
    def __init__(self, components, dtype=np.float64):
        self._arr = np.array(components, dtype=dtype)

    def __add__(self, v):
        return Vector(self._arr + v._arr)

    def __iadd__(self, v):
        self._arr += v._arr

    def __sub__(self, v):
        return Vector(self._arr - v._arr)

    def __isub__(self, v):
        self._arr -= v._arr

    def __imul__(self, c):
        if (not isinstance(c, (float, int))):
            raise TypeError("This operation is only supported for scalar "
                            "multiplication")

        self._arr *= c

    def __idiv__(self, c):
        if (not isinstance(c, (float, int))):
            raise TypeError("This operation is only supported for scalar "
                            "multiplication")

        self._arr /= c

    def __mul__(self, x):
        if (isinstance(x, Vector)):
            return (sum(x._arr * self._arr))
        elif (isinstance(x, (float, int))):
            return Vector(x * self._arr)
        else:
            raise TypeError("Right vector multiplication is only defined for "
                            "scalars and vectors")

    def __div__(self, x):
        if (isinstance(x, (float, int))):
            return Vector(self._arr / x)
        else:
            raise TypeError("Right vector division is only defined for "
                            "scalars.")

    def __rmul__(self, x):
        if (isinstance(x, Vector)):
            return (sum(x._arr * self._arr))
        elif (isinstance(x, (float, int))):
            return Vector(x * self._arr)
        elif (isinstance(x, np.array)):
            return Vector(np.dot(x, self._arr))
        else:
            raise TypeError("Left vector multiplication is only defined for "
                            "(numpy) matrices, vectors, or scalars.")

    def __neg__(self):
        return Vector(-self._arr)

    def __eq__(self, v):
        return np.array_equal(self._arr, v._arr)

    def __ne__(self, v):
        return not np.array_equal(self._arr, v._arr)

    def __str__(self):
        return "{}({})".format(type(self).__name__, str(self._arr.tolist()))

    def __repr__(self):
        return self.__str__()


class MetaStateVector(type):
    def __new__(cls, class_name, parents, attrs, params):
        def autoproperty(index):
            def getter(self):
                return self._arr[index]

            def setter(self, value):
                self._arr[index] = value

            return property(getter, setter)

        def autostr(self):
            lst = []

            for i in range(len(params)):
                lst.append("{}={}".format(self._params[i], self._arr[i * 2]))
                lst.append("{}1={}".format(self._params[i],
                                           self._arr[i * 2 + 1]))

            return "{}({})".format(type(self).__name__, str(lst))

        obj = super().__new__(cls, class_name, parents, attrs)
        obj._params = params
        obj.__str__ = autostr
        obj.__repr__ = autostr

        for i in range(len(params)):
            setattr(obj, params[i], autoproperty(i * 2))
            setattr(obj, "{}1".format(params[i]), autoproperty(i * 2 + 1))

        return obj


class PhysicalVector(Vector):
    def __init__(self, components=[0, 0, 0]):
        super().__init__(components)

        if (len(components) != 3 and len(components) != 2):
            raise ValueError("Physical vectors must be 2 or 3 dimensional")

        np.padding = 3 - len(components)
        self._arr = np.pad(components, (0, np.padding), 'constant')

    def __matmul__(self, v2):
        return PhysicalVector(np.cross(self._arr, v2._arr))

    def __rmatmul__(self, v2):
        return PhysicalVector(np.cross(v2, self._arr))

    @staticmethod
    def polar(r, phi):
        return PhysicalVector([r * np.cos(phi), r * np.sin(phi)])

    @staticmethod
    def spherical(r, theta, phi):
        return PhysicalVector([
            r * np.sin(theta) * np.cos(phi), r * np.sin(theta) * np.sin(phi),
            r * np.cos(theta)
        ])

    @staticmethod
    def cylindrical(r, phi, z):
        PhysicalVector([r * np.cos(phi), r * np.sin(phi), z])

    @staticmethod
    def normal(v):
        return PhysicalVector(v._arr / v.magnitude())

    def magnitude(self):
        return ((self._arr[0]**2 + self._arr[1]**2 + self._arr[2]**2)**(1 / 2))

    def normalize(self):
        self._arr /= self.magnitude()

    def rotate2(self, phi):
        A = np.array([[np.cos(phi), -np.sin(phi), 0],
                      [np.sin(phi), np.cos(phi), 0], [0, 0, 1]])

        self._arr = np.dot(A, self._arr)

    def rotate3(self, axis, phi):
        phi /= 2
        axis.normalize()

        w0 = np.cos(phi)
        x0 = axis.x * np.sin(phi)
        y0 = axis.y * np.sin(phi)
        z0 = axis.z * np.sin(phi)

        w1 = 0
        x1 = self.x
        y1 = self.y
        z1 = self.z

        w2 = -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0
        x2 = x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0
        y2 = -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0
        z2 = x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0

        w3 = np.cos(-phi)
        x3 = axis.x * np.sin(-phi)
        y3 = axis.y * np.sin(-phi)
        z3 = axis.z * np.sin(-phi)

        self.x = x3 * w2 + y3 * z0 - z3 * y2 + w3 * x2
        self.y = -x3 * z2 + y3 * w0 + z3 * x2 + w3 * y2
        self.z = x3 * y2 - y3 * x0 + z3 * w2 + w3 * z2

    @property
    def x(self):
        return self._arr[0]

    @property
    def y(self):
        return self._arr[1]

    @property
    def z(self):
        return self._arr[2]

    @property
    def r(self):
        return (self.x**2 + self.y**2)**(1 / 2)

    @property
    def rho(self):
        return self.magnitude()

    @property
    def phi(self):
        if (self._arr[0] == 0):
            if (self._arr[1] > 0):
                return np.pi / 2
            elif (self._arr[1] < 0):
                return -np.pi / 2
            else:
                return 0

        return np.arctan(self._arr[1] / self._arr[0])

    @property
    def theta(self):
        magnitude = self.magnitude()

        if (magnitude == 0):
            return 0

        return np.arccos(self._arr[2] / self.magnitude())

    @r.setter
    def r(self, value):
        theta = self.theta

        self._arr[0] = value * np.cos(theta)
        self._arr[1] = value * np.sin(theta)

    @phi.setter
    def phi(self, value):
        r = self.r

        self._arr[0] = r * np.cos(value)
        self._arr[1] = r * np.sin(value)

    @theta.setter
    def theta(self, value):
        phi = self.phi
        rho = self.rho

        self._arr[0] = rho * np.sin(value) * np.cos(phi)
        self._arr[1] = rho * np.sin(value) * np.sin(phi)
        self._arr[2] = rho * np.cos(value)

    @rho.setter
    def rho(self, value):
        phi = self.phi
        theta = self.theta

        self._arr[0] = value * np.sin(theta) * np.cos(phi)
        self._arr[1] = value * np.sin(theta) * np.sin(phi)
        self._arr[2] = value * np.cos(theta)

    @x.setter
    def x(self, value):
        self._arr[0] = value

    @y.setter
    def y(self, value):
        self._arr[1] = value

    @z.setter
    def z(self, value):
        self._arr[2] = value
