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
#include <cmath>
#include <Eigen/Dense>
#include <boost/multiprecision/gmp.hpp>

using std::abs; 
using boost::multiprecision::abs; 

/**
 * Given integers y and p, find a number t such that (y * t) mod p = 1,
 * using the extended Euclidean algorithm.
 */
template <int p>
int getMultiplicativeInverse(const int y)
{
    int t = 0; 
    int new_t = 1; 
    int r = p; 
    int new_r = y; 
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

        Fp()
        {
            this->value = 0; 
        }

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
                this->value = static_cast<int>(boost::multiprecision::numerator(x)) % p; 
            else    // x is not an integer and p is nonzero 
                throw std::runtime_error(
                    "Cannot assign non-integer value in field of finite "
                    "characteristic"
                );  
        }

        Fp(const Fp<p>& x)
        {
            this->value = x.value; 
        }

        ~Fp()
        {
        }

        // ---------------------------------------------------------- //
        //                    ASSIGNMENT OPERATORS                    //
        // ---------------------------------------------------------- //
        Fp<p>& operator=(const int& x)
        {
            if (p == 0)
                this->value = x;
            else
                this->value = x % p;
            return *this;  
        }
        
        Fp<p>& operator=(const Fp<p>& x)
        {
            this->value = x.value;
            return *this;  
        }

        // ---------------------------------------------------------- //
        //              EQUALITY AND INEQUALITY OPERATORS             //
        // ---------------------------------------------------------- //
        bool operator==(const int& x) const 
        {
            if (p == 0)
                return (this->value == x); 
            else
            { 
                int a = static_cast<int>(this->value); 
                return ((a - x) % p == 0);
            }
        }

        bool operator==(const boost::multiprecision::mpq_rational& x) const
        {
            if (p == 0)
            {
                return (this->value == x);
            } 
            else
            {
                if (boost::multiprecision::denominator(x) != 1)
                    return false;
                else  
                {
                    int a = static_cast<int>(this->value); 
                    int b = static_cast<int>(x); 
                    return ((a - b) % p == 0); 
                }
            } 
        }

        bool operator==(const Fp<p>& x) const
        {
            if (p == 0)
            {
                return (this->value == x.value);
            } 
            else 
            {
                int a = static_cast<int>(this->value); 
                int b = static_cast<int>(x.value); 
                return ((a - b) % p == 0);
            } 
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
        //                 GREATER-THAN AND LESS-THAN                 // 
        // ---------------------------------------------------------- //
        bool operator>(const Fp<p>& x) const 
        {
            return this->value > x.value; 
        }

        bool operator>=(const Fp<p>& x) const 
        {
            return this->value >= x.value; 
        }

        bool operator<(const Fp<p>& x) const 
        {
            return this->value < x.value; 
        }

        bool operator<=(const Fp<p>& x) const 
        {
            return this->value <= x.value; 
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
        //                           NEGATION                         //
        // ---------------------------------------------------------- //
        Fp<p> operator-() const 
        {
            return Fp<p>(-this->value); 
        }

        // ---------------------------------------------------------- //
        //                    ARITHMETIC OPERATORS                    //
        // ---------------------------------------------------------- //
        Fp<p>& operator+=(const int& y)
        {
            this->value += y; 
            if (p != 0)
                this->value = boost::multiprecision::mpq_rational(
                    static_cast<int>(this->value) % p
                ); 
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
                this->value = boost::multiprecision::mpq_rational(
                    static_cast<int>(this->value) % p
                ); 
            return *this; 
        }

        Fp<p>& operator+=(const Fp<p>& y)
        {
            this->value += y.value;
            if (p != 0)
                this->value = boost::multiprecision::mpq_rational(
                    static_cast<int>(this->value) % p
                ); 
            return *this;  
        }

        Fp<p>& operator-=(const int& y)
        {
            this->value -= y; 
            if (p != 0)
                this->value = boost::multiprecision::mpq_rational(
                    static_cast<int>(this->value) % p
                ); 
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
                this->value = boost::multiprecision::mpq_rational(
                    static_cast<int>(this->value) % p
                ); 
            return *this; 
        }

        Fp<p>& operator-=(const Fp<p>& y)
        {
            this->value -= y.value;
            if (p != 0)
                this->value = boost::multiprecision::mpq_rational(
                    static_cast<int>(this->value) % p
                ); 
            return *this; 
        }

        Fp<p>& operator*=(const int& y)
        {
            this->value *= y; 
            if (p != 0)
                this->value = boost::multiprecision::mpq_rational(
                    static_cast<int>(this->value) % p
                ); 
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
                this->value = boost::multiprecision::mpq_rational(
                    static_cast<int>(this->value) % p
                ); 
            return *this; 
        }

        Fp<p>& operator*=(const Fp<p>& y)
        {
            this->value *= y.value;
            if (p != 0)
                this->value = boost::multiprecision::mpq_rational(
                    static_cast<int>(this->value) % p
                ); 
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
                int t = getMultiplicativeInverse<p>(y % p); 
                this->value *= t;
                this->value = boost::multiprecision::mpq_rational(
                    static_cast<int>(this->value) % p
                ); 
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
                int t = getMultiplicativeInverse<p>(
                    static_cast<int>(boost::multiprecision::numerator(y) % p)
                ); 
                this->value *= t;
                this->value = boost::multiprecision::mpq_rational(
                    static_cast<int>(this->value) % p
                ); 
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
                int t = getMultiplicativeInverse<p>(
                    static_cast<int>(boost::multiprecision::numerator(y.value))
                ); 
                this->value *= t;
                this->value = boost::multiprecision::mpq_rational(
                    static_cast<int>(this->value) % p
                ); 
            }
            return *this;  
        } 
};

template <int p>
Fp<p> operator+(const Fp<p>& x, const int& y)
{
    return Fp<p>(x.value + y); 
}

template <int p>
Fp<p> operator+(const Fp<p>& x, const boost::multiprecision::mpq_rational& y)
{
    return Fp<p>(x.value + y); 
}

template <int p>
Fp<p> operator+(const int& x, const Fp<p>& y)
{
    return y + x; 
}

template <int p>
Fp<p> operator+(const boost::multiprecision::mpq_rational& x, const Fp<p>& y)
{
    return y + x; 
}

template <int p>
Fp<p> operator+(const Fp<p>& x, const Fp<p>& y)
{
    return Fp<p>(x.value + y.value); 
}

template <int p>
Fp<p> operator-(const Fp<p>& x, const int& y)
{
    return Fp<p>(x.value - y); 
}

template <int p>
Fp<p> operator-(const Fp<p>& x, const boost::multiprecision::mpq_rational& y)
{
    return Fp<p>(x.value - y); 
}

template <int p>
Fp<p> operator-(const int& x, const Fp<p>& y)
{
    return y - x; 
}

template <int p>
Fp<p> operator-(const boost::multiprecision::mpq_rational& x, const Fp<p>& y)
{
    return y - x; 
}

template <int p>
Fp<p> operator-(const Fp<p>& x, const Fp<p>& y)
{
    return Fp<p>(x.value - y.value); 
}

template <int p>
Fp<p> operator*(const Fp<p>& x, const int& y)
{
    return Fp<p>(x.value * y); 
}

template <int p>
Fp<p> operator*(const int& x, const Fp<p>& y)
{
    return y * x; 
}

template <int p>
Fp<p> operator*(const boost::multiprecision::mpq_rational& x, const Fp<p>& y)
{
    return y * x; 
}

template <int p>
Fp<p> operator*(const Fp<p>& x, const boost::multiprecision::mpq_rational& y)
{
    return Fp<p>(x.value * y); 
}

template <int p>
Fp<p> operator*(const Fp<p>& x, const Fp<p>& y)
{
    return Fp<p>(x.value * y.value); 
}

template <int p>
Fp<p> operator/(const Fp<p>& x, const int& y)
{
    // Check that y is nonzero 
    if (y == 0)
        throw std::runtime_error("Cannot divide by zero");

    // If p == 0, then simply return x / y
    if (p == 0)
    {
        return Fp<p>(x.value / y);
    }
    else    // Otherwise, multiply x by the multiplicative inverse of y
    {
        int t = getMultiplicativeInverse<p>(y % p); 
        return Fp<p>(x.value * t);
    } 
}

template <int p>
Fp<p> operator/(const Fp<p>& x, const boost::multiprecision::mpq_rational& y)
{
    if (p != 0 && boost::multiprecision::denominator(y) != 1)
        throw std::runtime_error(
            "Cannot divide number by non-integer number in field of "
            "finite characteristic"
        ); 

    // Check that y is nonzero 
    if (y == 0)
        throw std::runtime_error("Cannot divide by zero");

    // If p == 0, then simply return x / y
    if (p == 0)
    {
        return Fp<p>(x.value / y);
    }
    else    // Otherwise, multiply x by the multiplicative inverse of y
    {
        int t = getMultiplicativeInverse<p>(static_cast<int>(y) % p);
        return Fp<p>(x.value * t);
    } 
}

template <int p>
Fp<p> operator/(const Fp<p>& x, const Fp<p>& y)
{
    // Check that y is nonzero 
    if (y == 0)
        throw std::runtime_error("Cannot divide by zero");

    // If p == 0, then simply return x / y
    if (p == 0)
    {
        return Fp<p>(x.value / y.value);
    } 
    else    // Otherwise, multiply x by the multiplicative inverse of y
    {
        int t = getMultiplicativeInverse<p>(static_cast<int>(y.value));
        return Fp<p>(x.value * t);
    } 
}

template <int p>
std::ostream& operator<<(std::ostream& out, const Fp<p>& x)
{
    return out << x.value; 
}

template <int p>
Fp<p> abs(const Fp<p>& x)
{
    return Fp<p>(abs(x.value)); 
}

namespace Eigen {

template <>
struct NumTraits<Fp<0> > : NumTraits<boost::multiprecision::mpq_rational>
{
    typedef Fp<0> Real; 
    typedef Fp<0> NonInteger; 
    typedef Fp<0> Nested; 

    enum
    {
        IsComplex = 0, 
        IsInteger = 0,
        IsSigned = 1,
        RequireInitialization = 1,
        ReadCost = 5,    // TODO
        AddCost = 10,
        MulCost = 20
    };
};

template <>
struct NumTraits<Fp<2> > : NumTraits<int>
{
    typedef Fp<2> Real; 
    typedef Fp<2> NonInteger; 
    typedef Fp<2> Nested; 

    enum
    {
        IsComplex = 0, 
        IsInteger = 1,
        IsSigned = 0,
        RequireInitialization = 1,
        ReadCost = 1,
        AddCost = 3,
        MulCost = 3
    };
};

}   // namespace Eigen 

#endif
