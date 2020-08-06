import numpy as np  # type: ignore


class RangeError(ValueError):
    """Thrown when min is greater than or equal to max in a range."""
    pass


def clamp(limits, x):
    """
    A combination of max and min that ensures x is within the range defined.

    :returns: x under the clamp function.
    """
    if (len(limits) != 2):
        raise ValueError(
            "x_range must a list or tuple in the format (min, max)")
    if (limits[0] > limits[1]):
        raise RangeError("limits must be of form (min, max)")

    return max(limits[0], min(x, limits[1]))


def in_range(limits, x):
    """
    Tells whether x is within the specified range.

    :rtype: bool
    :returns: Whether or not x is within valid_range.
    """

    if (limits[0] >= limits[1]):
        raise RangeError("valid_range must be in the format (min, max)")
    return (x < limits[1] and x > limits[0])


def calc_1daffine_map(domain, image, strict=False):
    """
    Calculates an affine map (a map of the form mx + b).

    :returns: A lambda that encodes the map.
    """

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
    A = np.array(domain)
    B = np.array(image)
    T = np.linalg.inv(A) * B

    return (lambda x: T * x)


def cyclic_map(theta, domain):
    """
    Defines a cyclic domain (like angles between 0 and 2pi) to map theta to.
    For angles, this would map any angle out of the 0 to 2pi range to their
    equivalents in the 0 to 2pi range.

    :param x: The scalar to be mapped
    :param domain: A tuple of values expresnp.sing the domain as a range.

    :rtype: float
    :returns: the mapped theta.
    """
    if (domain[1] <= domain[0]):
        raise RangeError("The minimum must be less than the maximum.")

    gamma = domain[1] - domain[0]
    return (theta - gamma * ((theta - domain[0]) // gamma))


class Vector:
    """
    Defines an n-dimensional vector from an array with operator overloading
    that encodes dot product, scalar multiplication, and the cross product.
    """

    def __init__(self, components, dtype=np.float64):
        self._arr = np.array(components, dtype=dtype)

    def __add__(self, v):
        return type(self)(self._arr + v._arr)

    def __iadd__(self, v):
        self._arr += v._arr

    def __sub__(self, v):
        return type(self)(self._arr - v._arr)

    def __isub__(self, v):
        self._arr -= v._arr

    def __imul__(self, c):
        if (not isinstance(c, (float, int))):
            raise TypeError("This operation is only supported for scalar "
                            "multiplication")

        self._arr *= c

    def __getitem__(self, i):
        return self._arr[i]

    def __setitem__(self, i, val):
        self._arr[i] = val

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
    """
    Defines a vector with special index names for state space vectors.
    """

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


class CartesianVector(Vector):
    """
    Generates a 3D vector that encodes the dot product, scalar multiplication,
    and cross product into python operators.

    Also supports additional operations such as rotation.

    Encodes special names for the indicies of the vector.

    Supports cartesian, polar, and cylindrical notation. Every set operation
    sets r, theta, phi, x, y, and z simultaneously. For this reason, the most
    accurate representation will always be the one you use for construction.

    Supports 2D vectors (since they're a subset of 3D vectors)

    """

    def __init__(self, components=[0.0, 0.0, 0.0]):
        super().__init__(components, dtype=np.float64)

        if (len(components) != 3 and len(components) != 2):
            raise ValueError("Physical vectors must be 2 or 3 dimensional")

        np.padding = 3 - len(components)
        self._arr = np.pad(components, (0, np.padding), 'constant')

    def __matmul__(self, v2):
        return CartesianVector(np.cross(self._arr, v2._arr))

    def __rmatmul__(self, v2):
        return CartesianVector(np.cross(v2, self._arr))

    @staticmethod
    def polar(r, phi):
        """
        Constructs a polar style CartesianVector.
        """
        return CartesianVector([r * np.cos(phi), r * np.sin(phi)])

    @staticmethod
    def spherical(r, theta, phi):
        """
        Constructs a spherical style CartesianVector.
        """
        return CartesianVector([
            r * np.sin(theta) * np.cos(phi), r * np.sin(theta) * np.sin(phi),
            r * np.cos(theta)
        ])

    @staticmethod
    def cylindrical(r, phi, z):
        """
        Constructs a cylindrical style CartesianVector.
        """
        CartesianVector([r * np.cos(phi), r * np.sin(phi), z])

    @staticmethod
    def normal(v):
        """
        Constructs a normalized version of vector v.

        :param v: A CartesianVector to normalize

        :rtype: CartesianVector
        :returns: A CartesianVector that is the normalized form of v.
        """
        return CartesianVector(v._arr / v.magnitude())

    def magnitude(self):
        """
        Gets the magnitude of the vector.

        :rtype: float
        :returns: The magnitude of the vector
        """
        return ((self._arr[0]**2 + self._arr[1]**2 + self._arr[2]**2)**(1 / 2))

    def rotate2(self, phi):
        """
        Rotates the vector about the z axis in the xy plane.

        :param phi: Angle in radians to rotate counterclockwise.
        """
        A = np.array([[np.cos(phi), -np.sin(phi), 0.0],
                      [np.sin(phi), np.cos(phi), 0.0], [0.0, 0.0, 1.0]])

        self._arr = np.dot(A, self._arr)

    def rotate3(self, axis, phi):
        """
        Rotates the vector about an axis vector using quaternion
        multiplication.

        :param axis: The axis to rotate about.
        :param phi: Angle in radians to rotate counterclockwise.
        """
        phi /= 2
        axis.normalize()

        w0 = np.cos(phi)
        x0 = axis.x * np.sin(phi)
        y0 = axis.y * np.sin(phi)
        z0 = axis.z * np.sin(phi)

        w1 = 0.0
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

    @x.setter
    def x(self, value):
        self._arr[0] = value

    @property
    def y(self):
        return self._arr[1]

    @y.setter
    def y(self, value):
        self._arr[1] = value

    @property
    def z(self):
        return self._arr[2]

    @z.setter
    def z(self, value):
        self._arr[2] = value

    @property
    def r(self):
        return (self.x**2 + self.y**2)**(1 / 2)

    @r.setter
    def r(self, value):
        theta = self.theta

        self._arr[0] = value * np.cos(theta)
        self._arr[1] = value * np.sin(theta)

    @property
    def rho(self):
        return self.magnitude()

    @rho.setter
    def rho(self, value):
        phi = self.phi
        theta = self.theta

        self._arr[0] = value * np.sin(theta) * np.cos(phi)
        self._arr[1] = value * np.sin(theta) * np.sin(phi)
        self._arr[2] = value * np.cos(theta)

    @property
    def theta(self):
        magnitude = self.magnitude()

        if (magnitude == 0):
            return 0

        return np.arccos(self._arr[2] / self.magnitude())

    @theta.setter
    def theta(self, value):
        phi = self.phi
        rho = self.rho

        self._arr[0] = rho * np.sin(value) * np.cos(phi)
        self._arr[1] = rho * np.sin(value) * np.sin(phi)
        self._arr[2] = rho * np.cos(value)

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

    @phi.setter
    def phi(self, value):
        r = self.r

        self._arr[0] = r * np.cos(value)
        self._arr[1] = r * np.sin(value)
