import math

"""
    This exercisises is used to test understanding of Vectors. YOU are NOT to use any Numpy
    implementation for this exercises. 
"""

class Vector(object):
    """
        This class represents a vector of arbitrary size.
        You need to give the vector components. 
    """
    
    def __init__(self, components=None):
        """
            input: components or nothing
            simple constructor for init the vector
        """
        if components is None:
            components = []
        self.__components = list(components)


    def component(self, i):
        """
            input: index (start at 0)
            output: the i-th component of the vector.
        """
        if type(i) is int and -len(self.__components) <= i < len(self.__components):
            return self.__components[i]
        else:
            raise Exception("index out of range")

    def __len__(self):
        """
            returns the size of the vector
        """
        return len(self.__components)

    def modulus(self):
        """
            returns the euclidean length of the vector
        """
        summe = 0
        ## BEGIN SOLUTION
        for i in range(self.__len__()):
            summe += self.component(i)**2
        return math.sqrt(summe) ## EDIT THIS
        ## END SOLUTION

    def add(self, other):
        """
            input: other vector
            assumes: other vector has the same size
            returns a new vector that represents the sum.
        """
        size = len(self.__components)
        if size == len(other):
            ## BEGIN SOUTION
            result=[self.__components[i]+other.component(i) for i in range(size)]
            return  Vector(result)## EDIT THIS
            ## END SOLUTION
        else:
            raise Exception("must have the same size")

    def sub(self, other):
        """
            input: other vector
            assumes: other vector has the same size
            returns a new vector that represents the difference.
        """
        size = len(self)
        if size == len(other):
            ## BEGIN SOUTION
            result=[self.__components[i]-other.component(i) for i in range(size)]
            return Vector(result)  ## EDIT THIS
            ## END SOLUTION
        else:  # error case
            raise Exception("must have the same size")

    def multiply(self, other):
        """
            multiply implements the scalar multiplication 
            and the dot-product
        """
        if isinstance(other, float) or isinstance(other, int): #scalar multiplication
            ## BEGIN SOLUTION
            result=[self.__components[i]*other for i in range(self.__len__())]
            return Vector(result) ## EDIT THIS
            ## END SOLUTION
        elif isinstance(other, Vector) and (len(self) == len(other)): # dot product
            size = len(self)
            summe = 0
            ## BEGIN SOLUTION
            for i in range(size):
                summe += self.__components[i]*other.component(i)
            return summe
            ## END SOLUTION
        else:  # error case
            raise Exception("invalid operand!")

    
    def scalar_proj(self, other):
        """ 
            Computes the scalar projection of vector r on s.
        """

        ### BEGIN SOLUTION
        return self.multiply(other)/other.modulus() ## EDIT THIS
        ### END SOLUTION
        
    def vector_proj(self, other):
        """ 
            Computes the vector projection of vector r on s.
            use the other functions created above.
        """
    
        ### BEGIN SOLUTION
        return other.multiply(self.multiply(other)/other.modulus()**2) ## EDIT THIS
        ### END SOLUTION

#print()
#print("Please, uncomment the following to see the results for some examples.")

import unittest

class Test(unittest.TestCase):
    def test_component(self):
        """
            test for method component
        """
        x = Vector([1, 2, 3])
        self.assertEqual(x.component(0), 1)
        self.assertEqual(x.component(2), 3)
        try:
            y = Vector()
            self.assertTrue(False)
        except:
            self.assertTrue(True)


    def test_size(self):
        """
            test for size()-method
        """
        x = Vector([1, 2, 3, 4])
        self.assertEqual(len(x), 4)

    def test_modulus(self):
        """
            test for the eulidean length
        """
        x = Vector([1, 2])
        self.assertAlmostEqual(x.modulus(), 2.236, 3)

    def test_add(self):
        """
            test for + operator
        """
        x = Vector([1, 2, 3])
        y = Vector([1, 1, 1])
        self.assertEqual((x.add(y)).component(0), 2)
        self.assertEqual((x.add(y)).component(1), 3)
        self.assertEqual((x.add(y)).component(2), 4)

    def test_sub(self):
        """
            test for - operator
        """
        x = Vector([1, 2, 3])
        y = Vector([1, 1, 1])
        self.assertEqual((x.sub(y)).component(0), 0)
        self.assertEqual((x.sub(y)).component(1), 1)
        self.assertEqual((x.sub(y)).component(2), 2)

    def test_multiply(self):
        """
            test for vector multiplication
        """
        x = Vector([1, 2, 3])
        a = Vector([2, -1, 4])  # for test of dot-product
        b = Vector([1, -2, -1])
        self.assertEqual((x.multiply(3.0)).component(0),3 )
        self.assertEqual((a.multiply(b)), 0)

    def test_scalar_projection(self):
        """
            test for scalar projection
        """
        x = Vector([3, 4])
        y = Vector([4, 3])
        self.assertEqual(x.scalar_proj(y), 4.8)
        
    def test_vector_projection(self):
        """
            test for scalar projection
        """
        x = Vector([3, 4])
        y = Vector([4, 3])
        self.assertEqual((y.vector_proj(x)).component(1), 3.84)
        
        

if __name__ == "__main__":
    unittest.main()

#V1=Vector([1, -2, 7])
#V2=Vector([3, 5, 0])
#V3=Vector([-3., 0.0, 4.0])
#
#print('Modulus, V1: {}'.format(V1.modulus()))
#print('Modulus, V2: {}'.format(V2.modulus()))
#print('Modulus, V3: {}'.format(V3.modulus()))
#print()
#
#print("Scalar product of V3 by 2: ", [V3.multiply(2).component(i) for i in range(V3.__len__())])
#print()
#
#print('dot product of V1 and V2:', V1.multiply(V2))
#print('dot product of V1 and V3:', V1.multiply(V3))
#print('dot product of V2 and V3:', V3.multiply(V2))
#print()
#
#print("V1+V2: ", [V1.add(V2).component(i) for i in range(V1.__len__())])
#print("Scalar_projection of V1 onto V2: ",V1.scalar_proj(V2))
#print("Vector_projection of V1 onto V2: ", [V1.vector_proj(V2).component(i) for i in range(V1.__len__())])
#print(V1.vector_proj(V2))