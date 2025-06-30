/**
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     6/30/2025
 */
#ifndef FIELDS_OF_ARBITRARY_CHARACTERISTIC_HPP
#define FIELDS_OF_ARBITRARY_CHARACTERISTIC_HPP

#include <iostream>
#include <Eigen/Dense>
#include <boost/multiprecision/gmp.hpp>

/**
 * Given integers y and p, find a number t such that (y * t) mod p = 1,
 * using the extended Euclidean algorithm.
 */
template <int p>
int getMultiplicativeInverse(const int y)
{
    int y_ = static_cast<int>(y.value);
    int t = 0; 
    int new_t = 1; 
    int r = p; 
    int new_r = y_; 
    while (new_r != 0)
    {
        int quotient = r / new_r; 
        int next_r = r - quotient * new_r; 
        int next_t = t - quotient * new_t;
        r = new_r; 
        new_r = next_r;  
        t = new_t;
        new_t = next_t;  
    }
    if (t < 0)
        t += p;

    return t; 
}

/**
 * Fields of characteristic p. 
 */
template <int p>
class Fp
{
    public:
        boost::multiprecision::mpq_rational value;

        Fp(const int x)
        {
            if (p == 0)
                this->value = static_cast<boost::multiprecision::mpq_rational>(x);
            else 
                this->value = static_cast<boost::multiprecision::mpq_rational>(x % p);  
        }

        Fp(const boost::multiprecision::mpq_rational x)
        {
            if (p == 0)
                this->value = x; 
            else if (boost::multiprecision::denominator(x) == 1)
                this->value = x; 
            else    // x is not an integer and p is nonzero 
                throw std::runtime_error(
                    "Cannot assign non-integer value in field of finite "
                    "characteristic"
                );  
        }

        ~Fp()
        {
        }

        // ---------------------------------------------------------- //
        //              EQUALITY AND INEQUALITY OPERATORS             //
        // ---------------------------------------------------------- //
        bool operator==(const int& x) const 
        {
            if (p == 0)
                return (this->value == x); 
            else 
                return ((this->value - x) % p == 0); 
        }

        bool operator==(const boost::multiprecision::mpq_rational& x) const
        {
            if (p == 0)
                return (this->value == x); 
            else 
                return ((this->value - x) % p == 0); 
        }

        bool operator==(const Fp<p>& x) const
        {
            if (p == 0)
                return (this->value == x.value); 
            else 
                return ((this->value - x.value) % p == 0); 
        }

        bool operator!=(const int& x) const 
        {
            return !(*this == x); 
        }

        bool operator!=(const boost::multiprecision::mpq_rational& x) const 
        {
            return !(*this == x); 
        }

        bool operator!=(const Fp<p>& x) const 
        {
            return !(*this == x); 
        }

        // ---------------------------------------------------------- //
        //                  INCREMENT AND DECREMENT                   //
        // ---------------------------------------------------------- //
        Fp<p>& operator++()
        {
            this->value++; 
            if (p != 0 && this->value == p)
                this->value = 0; 
            return *this; 
        }
        
        Fp<p>& operator--()
        {
            this->value--;
            if (p != 0 && this->value == -1)
                this->value = p - 1;
            return *this;  
        }

        // ---------------------------------------------------------- //
        //                    ARITHMETIC OPERATORS                    //
        // ---------------------------------------------------------- //
        Fp<p>& operator+=(const int& y)
        {
            this->value += y; 
            if (p != 0)
                this->value %= p; 
            return *this; 
        }

        Fp<p>& operator+=(const boost::multiprecision::mpq_rational& y)
        {
            #ifdef CHECK_FP_RATIONAL_NONINTEGER
                if (p != 0 && boost::multiprecision::denominator(y) != 1)
                    throw std::runtime_error(
                        "Cannot add non-integer number to number in field of "
                        "finite characteristic"
                    ); 
            #endif
            this->value += y; 
            if (p != 0)
                this->value %= p; 
            return *this; 
        }

        Fp<p>& operator+=(const Fp<p>& y)
        {
            this->value += y.value;
            if (p != 0)
                this->value %= p;
            return *this;  
        }

        Fp<p>& operator-=(const int& y)
        {
            this->value -= y; 
            if (p != 0)
                this->value %= p; 
            return *this; 
        }

        Fp<p>& operator-=(const boost::multiprecision::mpq_rational& y)
        {
            #ifdef CHECK_FP_RATIONAL_NONINTEGER
                if (p != 0 && boost::multiprecision::denominator(y) != 1)
                    throw std::runtime_error(
                        "Cannot subtract non-integer number from number in "
                        "field of finite characteristic"
                    ); 
            #endif
            this->value -= y; 
            if (p != 0)
                this->value %= p; 
            return *this; 
        }

        Fp<p>& operator-=(const Fp<p>& y)
        {
            this->value -= y.value;
            if (p != 0)
                this->value %= p;  
            return *this; 
        }

        Fp<p>& operator*=(const int& y)
        {
            this->value *= y; 
            if (p != 0)
                this->value %= p; 
            return *this; 
        }

        Fp<p>& operator*=(const boost::multiprecision::mpq_rational& y)
        {
            #ifdef CHECK_FP_RATIONAL_NONINTEGER
                if (p != 0 && boost::multiprecision::denominator(y) != 1)
                    throw std::runtime_error(
                        "Cannot subtract non-integer number from number in "
                        "field of finite characteristic"
                    ); 
            #endif
            this->value *= y; 
            if (p != 0)
                this->value %= p; 
            return *this; 
        }

        Fp<p>& operator*=(const Fp<p>& y)
        {
            this->value *= y.value;
            if (p != 0)
                this->value %= p;
            return *this;  
        }

        Fp<p>& operator/=(const int& y)
        {
            // Check that y is nonzero 
            if (y == 0)
                throw std::runtime_error("Cannot divide by zero");

            // If p == 0, then simply return x / y
            if (p == 0)
            {
                this->value /= y;
            }
            else    // Otherwise, multiply x by the multiplicative inverse of y
            {
                int t = getMultiplicativeInverse<p>(y); 
                this->value *= t;
                this->value %= p;
            }
            return *this;  
        }

        Fp<p>& operator/=(const boost::multiprecision::mpq_rational& y)
        {
            #ifdef CHECK_FP_RATIONAL_NONINTEGER
                if (p != 0 && boost::multiprecision::denominator(y) != 1)
                    throw std::runtime_error(
                        "Cannot subtract non-integer number from number in "
                        "field of finite characteristic"
                    ); 
            #endif
            
            // Check that y is nonzero
            if (y == 0)
                throw std::runtime_error("Cannot divide by zero");

            // If p == 0, then simply return x / y
            if (p == 0)
            {
                this->value /= y;
            }
            else    // Otherwise, multiply x by the multiplicative inverse of y
            {
                int t = getMultiplicativeInverse<p>(boost::multiprecision::numerator(y)); 
                this->value *= t;
                this->value %= p;
            }
            return *this;  
        }

        Fp<p>& operator/=(const Fp<p>& y)
        {
            // Check that y is nonzero 
            if (y == 0)
                throw std::runtime_error("Cannot divide by zero");

            // If p == 0, then simply return x / y
            if (p == 0)
            {
                this->value /= y.value;
            }
            else    // Otherwise, multiply x by the multiplicative inverse of y
            {
                int t = getMultiplicativeInverse<p>(y.value); 
                this->value *= t;
                this->value %= p;
            }
            return *this;  
        } 
};

Fp<p> operator+(const Fp<p>& x, const Fp<p>& y) const 
{
    return Fp<p>(x.value + y.value); 
}

Fp<p> operator-(const Fp<p>& x, const Fp<p>& y) const
{
    return Fp<p>(x.value - y.value); 
}

Fp<p> operator*(const Fp<p>& x, const Fp<p>& y)
{
    return Fp<p>(x.value * y.value); 
}

Fp<p> operator/(const Fp<p>& x, const Fp<p>& y)
{
    // Check that y is nonzero 
    if (y == 0)
        throw std::runtime_error("Cannot divide by zero");

    // If p == 0, then simply return x / y
    if (p == 0)
        return Fp<p>(x.value / y.value);  

    // Otherwise, multiply x by the multiplicative inverse of y 
    int t = getMultiplicativeInverse<p>(y.value); 
    return Fp<p>(x_ * t);  
}

#endif
