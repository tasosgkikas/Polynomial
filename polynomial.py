# from fourier import fft as fft, ifft as ifft
import re

import numpy as np
from numpy.fft import fft, ifft
# from matplotlib import pyplot as plt


class Polynomial:
    """
    Class for polynomials
    Each polynomial is defined by a given string in terms of a variable,
    raised to integer powers and with float coefficients

    self._array[k] is the coeff of x^k
    """
    list = []

    def __init__(self, pol: str = '0', var: str = 'x'):
        self._pol = pol.replace(' ', '')  # this is shown
        self._var = var
        self._deg = 0
        # pattern: [+-][numbers]x^[numbers] -> ([+-]?\d*)(_var?\^?\d*)
        self._pattern = r'([+-]?\d*\.?\d*)(' + self._var + r'?\^?\d*)'
        self._array = []  # used for computations

        self._mkArray()
        Polynomial.list.append(self)

    def copy(self) -> 'Polynomial':
        return Polynomial.fromArray(self._array)

    @staticmethod
    def fromArray(arr: (list | np.ndarray), var: str = 'x') -> 'Polynomial':
        new = Polynomial()
        # print(new._pol, new._deg, new._array)
        new._var = var
        new._array = arr
        new._deg = len(arr) - 1
        new._reformat()
        return new

    def __str__(self):
        return self._pol

    def _mkArray(self):
        # retrieving auxiliary matrix
        matrix = self._mkMatrix()
        self._deg = max(term[1] for term in matrix)
        '''
        constructing _array of size _deg+1
        (+1 because 0th power is also included)
        by assigning to index k
        the sum of the coeffs of power k
        '''
        self._array = np.zeros(self._deg + 1)
        for k in range(self._deg + 1):
            self._array[k] = sum(term[0] for term in matrix if term[1] == k)

        self._reformat()

    def _mkMatrix(self) -> list:
        # fixing double signs
        signs = {'+-': '-', '-+': '-', '++': '+', '--': '+'}
        for mix in signs:
            self._pol = self._pol.replace(mix, signs[mix], self._pol.count(mix))

        matrix = re.findall(self._pattern, self._pol)
        matrix.remove(('', ''))

        # converting tuples of matrix to lists
        matrix = [list(term) for term in matrix]

        for i in range(len(matrix)):
            # fixing coeffs with value 1
            if matrix[i][0] in ['', '+']:
                matrix[i][0] = 1
            elif matrix[i][0] == '-':
                matrix[i][0] = -1
            elif matrix[i][0].count('.') == 1:
                matrix[i][0] = float(matrix[i][0])
            else:
                matrix[i][0] = int(matrix[i][0])

            # fixing powers
            if self._var not in matrix[i][1]:  # power = 0
                matrix[i][1] = 0
            elif '^' not in matrix[i][1]:  # power = 1
                matrix[i][1] = 1
            else:  # power >= 2
                matrix[i][1] = int(matrix[i][1].replace(self._var + '^', ''))

        return matrix

    def _reformat(self):
        if self._array[-1] == 0:
            self._pol = ' 0'
        else:
            self._pol = ''
        for p in range(self._deg, -1, -1):
            coeff = self._array[p]
            if coeff:
                if str(coeff)[-1] == '.' or \
                        str(coeff)[-2:] == '.0':
                    coeff = int(coeff)
                if len(self._pol) and coeff > 0:
                    self._pol += '+'
                if coeff == -1 and p != 0:
                    self._pol += '-'
                elif coeff != 1 or p == 0:
                    self._pol += str(coeff)
                if p > 0:
                    self._pol += self._var
                if p > 1:
                    self._pol += '^' + str(p)
                self._pol += ' '

    def details(self):
        print()
        print(self)
        print('Degree:', self._deg)
        print('Array:', self._array)

    def __add__(self, other) -> 'Polynomial':
        if type(other) != Polynomial:
            other = Polynomial(str(other))
        return Polynomial(self._pol + '+' + other._pol)

    def __radd__(self, other) -> 'Polynomial':
        return self + other

    def __iadd__(self, other) -> 'Polynomial':
        return self + other

    def __neg__(self) -> 'Polynomial':
        return Polynomial.fromArray([-self._array[p] for p in range(len(self._array))])

    def __sub__(self, other) -> 'Polynomial':
        return self + (-other)

    def __rsub__(self, other) -> 'Polynomial':
        return -(self - other)

    def __isub__(self, other) -> 'Polynomial':
        return self - other

    def __mul__(self, other) -> 'Polynomial':
        if type(other) != Polynomial:
            other = Polynomial(str(other))
        with_fft = False
        if with_fft:  # doesn't work fine
            n = self._deg + other._deg + 1
            self_fft = np.array(round(x) for x in fft(self._array, n))
            other_fft = np.array(round(x) for x in fft(other._array, n))
            prod = self_fft * other_fft
            output = ifft(prod).real
            # output = ifft(fft(self._array)*fft(other._array)).real
        else:  # standard method, takes O(n^2) time
            output = np.zeros(self._deg + other._deg + 1)
            ij = [(x, y) for x in range(self._deg + 1) for y in range(other._deg + 1)]
            for i, j in ij:
                output[i + j] += self._array[i] * other._array[j]
        return Polynomial.fromArray(output)

    def __rmul__(self, other) -> 'Polynomial':
        if type(other) != Polynomial:
            other = Polynomial(str(other))
        return self * other

    def __imul__(self, other):
        return self * other

    def __pow__(self, power, modulo=None) -> 'Polynomial':
        result = Polynomial('1')
        for k in range(power):
            result *= self
        return result

    def __eq__(self, other) -> bool:
        if type(other) != Polynomial:
            other = Polynomial(str(other))
        if self._deg != other._deg:
            return False
        for i in range(len(self._array)):
            if self._array[i] != other._array[i]:
                return False
        return True

    def __ne__(self, other) -> bool:
        return not self == other

    def __call__(self, x=None):
        if x is None:
            output = self
        elif type(x) in [float, int, complex,
                         np.float32, np.float64,
                         np.int32, np.int64]:
            output = 0
            for power, coeff in enumerate(self._array):
                output += coeff * x ** power
        elif type(x) in [list, tuple, np.ndarray]:
            output = []
            for _x in x:
                output.append(self(_x))
        else:
            raise TypeError(
                "'x' must be either a number"
                " (of type float/in/complex)"
                ", an _array (of type list/"
                "tuple/numpy.ndarray), or None")
        return output

    def solve(self, equalto: (float | int | 'Polynomial') = 0, digits: int = 2):
        if str(equalto).replace('.', '', 1).isdigit():
            equalto = Polynomial(str(equalto))
        if type(equalto) != Polynomial:
            raise TypeError('"equalto" must be either a number'
                            'or a Polynomial class object')
        p = self - equalto
        return p._newton(0, digits)

    def derivative(self, x=None, order: int = 1):
        if order == 0:
            return self(x)
        else:
            if self._deg == 0:
                der = [0]
            else:
                der = [k * c for k, c in enumerate(self._array) if k != 0]
            p = Polynomial.fromArray(der)
            return p.derivative(x, order - 1)

    _count = 0  # used at _newton

    def _newton(self, x: (float | int), digits):
        Polynomial._count += 1
        print(Polynomial._count, 'x =', x)
        precision = 10 ** (-digits)
        while self.derivative(x) == 0:
            x += precision
        dx = self(x) / self.derivative(x)
        if abs(dx) < precision:
            return x
        else:
            return self._newton(x - dx, digits)

    def companion_matrix(self):
        zeros = np.zeros(self._deg - 1)
        ones = np.ones([self._deg - 1, self._deg - 1])
        coeffs = [-c / self._array[-1] for c in self._array[:-1]]

        first = np.concatenate([zeros, ones], axis=1)
        print(zeros)
        last = np.concatenate([first, coeffs], axis=0)

        return last

    @staticmethod
    def showList():
        print('\nCurrent polynomials:')
        for i, pol in enumerate(Polynomial.list):
            print(str(i + 1) + '.', pol)
        print()


def main():
    """
    1. Extract the 2 coefficient matrices
    2. Multiply their fft's
    3. Get the coefficient matrix by ifft-ing the product
    """

    # print('Give 2 polynomials:')
    # pols = [input('P(x) = '), input('Q(x) = ')]

    # pols = ['  2x^4 + 4x^3 -   x^2+ 7  ',
    #         '-7+x^3+7-5x^2-3',
    #         'x^2+1',
    #         '-5x^4-.4x']
    #
    # pols = [Polynomial(p) for p in pols]
    #
    # p = pols[2]
    # q = pols[1]
    # print(p, q, p-q, sep='\n')
    # Polynomial.showList()

    # p = Polynomial('x^3-x-5')
    # q = Polynomial('x^3+3')
    # Polynomial.showList()
    # print(p*q)
    # print('x^6-x^4-2x^3-3x-15')

    # p = Polynomial('-2x^3+5x^2')
    # q = Polynomial('x^3+3x^2-1')
    # print(q*p)
    # print(p('d'))
    # x = np.linspace(-5, 5, 100)
    #
    # plt.plot(x, p(x))
    # plt.show()
    # s = p.solve()
    # print(p.derivative(1)(1))

    # print(Polynomial('x^2-x-1').solve(5))

    # q = Polynomial('x^3')
    # q = ((7-3*q**2).derivative(order=3)*(q**3-2*q))
    # print(q(2-1j))
    # Polynomial.showList()

    # print(Polynomial('x-1')**4)
    print(Polynomial('x^3-2x^2-5x+6').companion_matrix())


if __name__ == '__main__' or True:
    main()
