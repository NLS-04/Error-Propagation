"""
# Error-Propagation

This Scripts is capable of calculating the *Gaussian Error Propagation* and create simple statistics for datasets.

This was created out of the frustration with the module *Fehlerrechnung* (engl. Error-Propagation) at my university and the ability to reliably cross check my results.

### INFORMATION FOR END USERS

Although this library is built around the concept of accepting *all* numeric data types (`float`, `int`, `Decimal`), it is **highly encouraged** to use strings (`str`) to represent numbers. This recommendation stems from the inherent rounding errors associated with floating-point data types. The library performs all calculations using the built-in `decimal` module, ensuring the best accuracy when using string inputs.

In the following documentation `number_like` refers to `Decimal | int | float | str`

- ### Math functions 
(all trigonometric functions currently only support arguments in `radians`)

argument `x` is of type ` Value | number_like`. All following functions return a `Value` instance

  - `exp( x )`
  - `log( argument [, base:x defaults to Euler-e ] )`
  - `sin( x )`
  - `cos( x )`
  - `tan( x )`
  - `arcsin( x )`
  - `arccos( x )`
  - `arctan( x )`
  - `sinh( x )`
  - `cosh( x )`
  - `tanh( x )`
  - `arsinh( x )`
  - `arcosh( x )`
  - `artanh( x )`


Following classes and functions that are relevant for you:

- ### Classes
  - ### Value
    - ```py
        Value(
            best_value: number_like,
            error: number_like
            [, scale_exponent: int=0 ]
            [, precision: int=2 ]
            [, id:str ] 
        )
        ```
    -   | operation        | symbol | example              |
        | :--------------: | :----: | -------------------- |
        | negate           | `-`    | `-a`                 |
        | positive         | `+`    | `+a`                 |
        | addition         | `+`    | `a+b` `1+a` `a+1`    |
        | subtraction      | `-`    | `a-b` `2-a` `a-2`    |
        | multiplication   | `*`    | `a*b` `3*a` `a*3`    |
        | division         | `/`    | `a/b` `4/a` `a/4`    |
        | exponentiation   | `**`   | `a**b` `5**a` `a**5` |
        | Compare Equal    | `==`   | `a==b` `6==a` `a==6` |
        | Compare Unequal  | `!=`   | `a!=b` `7!=a` `a!=7` |


  - ### Measurement
    - ```py
      Measurement( 
          data: Sequence[number_like]
          [, scale_exponent: int=0]
          [, precision: int=2 ]
          [, id:str ]
      )
      ```
  - ### Linear Regression
    The following classes assume that *only* the `Y`-Component of the (`X`, `Y`)-Point-cloud is *erroneous*.

    - simple linear Regression (all `Y`-Values have the same absolute error)
      ```py
      Regression_linear_simple(
        x_data: Sequence[number_like],
        y_data: Sequence[number_like]
      )
      ```

    - *general* linear Regression (`Y`-Values can have different absolute errors)
      ```py
      Regression_linear(
        x_data:Sequence[number_like],
        y_data:Sequence[Value]
      )
      ```

  - ### PF [optional for advanced use]
    Converting data from one *unit-scale* to an other specific *unit-scale*.

    *PF* employs *Decimal-Prefixes* or *Pre-factors* to denote the magnitude or scale of units( i.e *kilometers* (*km*), *microfarad* (*µF*) etc ).
    
    example:
    ```py
    >>> # creating a mass `Value` in units of milligram that are converted to kilograms
    >>> Value( '100', '3.5', exp=PF.m.to(PF.k) )
    (100 ± 3.5)e-6    δe= 0.035
    
    >>> # creating a distance `Value` in units of kilometers that are converted to meters
    >>> Value( '3.21', '0.11', exp=PF.k.to(PF.NONE) )
    (3.21 ± 0.11)e+3    δe= 0.034
    ```

    - predefined class-variables

        | name    | alias | factor |
        | ------- | ----- | ------ |
        | `PETA`  | `P`   | `E+15` |
        | `TERA`  | `T`   | `E+12` |
        | `GIGA`  | `G`   | `E+9 ` |
        | `MEGA`  | `M`   | `E+6 ` |
        | `KILO`  | `k`   | `E+3 ` |
        | `HECTO` | `h`   | `E+2 ` |
        | `DECA`  | `da`  | `E+1 ` |
        | `NONE`  | `_`   | `1   ` |
        | `DEZI`  | `d`   | `E-1 ` |
        | `CENTI` | `c`   | `E-2 ` |
        | `MILLI` | `m`   | `E-3 ` |
        | `MICRO` | `µ`   | `E-6 ` |
        | `NANO`  | `n`   | `E-9 ` |
        | `PICO`  | `p`   | `E-12` |
        | `FEMTO` | `f`   | `E-15` |

### Example Code
```py
>>> # creating statistics to a set of measurements, the supplied data is scaled by the Pre-factor `PF.m` same as `exp=-3`
>>> length = Measurement( [1.249, 1.234, 1.252, 1.238, 1.235, 1.246, 1.262, 1.255, 1.243], PF.m, id='d' ) # data is in millimeter
Measurement:
+----------------------------------------------------+
|         N:              9                          |
| Spanwidth:        0.028 * 10^-3                    |
| -------------------------------------------------- |
|   Average:       (1246 ± 3.2)e-6       δe= 0.0025  |
|    Median:        (1.2 ± 0)e-3         δe= 0E+27   |
| -------------------------------------------------- |
|        σ²:        (80 ± 4.4)e-9        δe= 0.056   |
|         σ:       (8.9 ± 2.1)e-6        δe= 0.24    |
|        s²:        (90 ± 5.6)e-9        δe= 0.062   |
|         s:       (9.5 ± 2.4)e-6        δe= 0.25    |
+----------------------------------------------------+

>>> # defining erroneous values
>>> # id is optional and does not alter results of your calculations
>>> # it is mainly there to be a placeholder in the .print_info_equation() output (example is below)
>>> a = Value( '1.630' , '0.021' , prec=2, id='a' )
>>> b = Value( '0.6649', '0.0040', prec=3, id='b' )
>>> x = Value( '1.376' , '0.037' , prec=2, id='x' )

>>> a.v # alias for a.value
1.63

>>> a.error # alias for a.e
0.021

>>> a.re # alias for a.relative_error
0.013

>>> # examples displaying the syntax of different calculations
>>> y1 = 13*a*x + 14*a*b*x**2 + 21*a*b**3
67.947 ± 2.513    δe= 0.03699

>>> y2 = exp( ( a - x ) / x )
(1202.73 ± 42.48)e-3    δe= 0.03532

>>> y3 = b * sin( a * x )
(520.3 ± 27.86)e-3    δe= 0.05355

>>> y4 = ( x - a ) / ( x + b )
(-124.45 ± 22.84)e-3    δe= 0.1835

>>> y1.print_info_equation()
Equation   : b**3 * a * 21 + x * a * 13 + x**2 * b * a * 14
Derivatives:
        x:      a * 13 + b * a * x * 28
        b:      a * b**2 * 63 + x**2 * a * 14
        a:      b**3 * 21 + x * 13 + x**2 * b * 14
"""



from __future__ import annotations

from decimal import Decimal
from typing import Optional, Sequence, Any, Callable
from functools import cache, reduce

from decimal import *

import math
from enum import Enum, auto

getcontext().rounding = ROUND_HALF_UP

PRINT_MATH_DEBUG = False

EPSILON_EXP = 12

def is_integer( d1:Decimal ) -> bool:
    epsilon = EPSILON_EXP
    
    while True:
        try:
            return d1.to_integral_value().compare( round(d1, epsilon) ) == Decimal('0')
        except InvalidOperation:
            epsilon -= 1
        finally:
            if epsilon <= 0:
                return True

#--------------------------------------------------------------------------------------------------
# Enums
#--------------------------------------------------------------------------------------------------
class Print_mode(Enum):
    """
    Determine whether the relative error should be displayed when calling the str(...) method on a Value instance

    ---
    enums:
    - `WITH_RELATIVE_ERROR`: appends the relative error at the end
    - `NO_RELATIVE_ERROR`: does not append the relative error
    """
    WITH_RELATIVE_ERROR = auto()
    NO_RELATIVE_ERROR = auto()
    
    DEFAULT = WITH_RELATIVE_ERROR

class Precision_mode(Enum):
    """
    Determine which precision should be used on operations involving multiple Value instances with different precision settings

    ---
    enums:
    - `TOWARDS_MAX`: favours the larger precision
    - `TOWARDS_MIN`: favours the smaller precision
    """
    TOWARDS_MAX = auto()
    TOWARDS_MIN = auto()
    
    DEFAULT = TOWARDS_MAX

class Trigonometry_mode(Enum):
    """
    Indicate how arguments to trigonometric methods are interpreted as of their units

    ---
    enums:
    - `DEGREES`: arguments are given in degrees
    - `RADIANS`: arguments are given in radians
    """
    DEGREES = 180.0 / math.pi
    RADIANS = 1.0 / DEGREES
    
    DEFAULT = DEGREES

class Propagation_mode(Enum):
    """
    Determine which precision should be used on operations involving multiple Value instances with different precision settings
    Determine how the error between Values should be calculated.
    
    ## ATTENTION
    #### Currently only `GAUSSIAN_ERROR` is implemented
    #### Currently `GREATES_ERROR` and `DEFAULT` are interpreted as `GAUSSIAN_ERROR`
    
    ---
    enums:
    - `GREATES_ERROR`: favours the larger precision
    - `GAUSSIAN_ERROR`: favours the smaller precision
    """ 
    GREATES_ERROR = auto()
    GAUSSIAN_ERROR = auto()
    
    DEFAULT = GAUSSIAN_ERROR

class PF():
    """
    Converting data from one *unit-scale* to an other specific *unit-scale*.

    *PF* employs *Decimal-Prefixes* or *Pre-factors* to denote the magnitude or scale of units( i.e *kilometer* (*km*), *microfarad* (*µF*) etc ).
    
    example:
    >>> # creating a mass `Value` in units of milligram that are converted to kilograms
    >>> Value( '100', '3.5', exp=PF.m.to(PF.k) )
    (100 ± 3.5)e-6    δe= 0.035
    
    >>> # creating a distance `Value` in units of kilometers that are converted to meters
    >>> Value( '3.21', '0.11', exp=PF.k.to(PF.NONE) )
    (3.21 ± 0.11)e+3    δe= 0.034
    """
    
    # cspell: ignore PETA GIGA HECTO DEKA DEZI CENTI MILLI PIKO FEMTO
    
    @classmethod
    @property
    def PETA(cls) -> PF:
        '''Pre-factor of `10**15`'''
        return PF( 15 )
    
    @classmethod
    @property
    def TERA(cls) -> PF:
        '''Pre-factor of `10**12`'''
        return PF( 12 )
    
    @classmethod
    @property
    def GIGA(cls) -> PF:
        '''Pre-factor of `10**9`'''
        return PF( 9 )
    
    @classmethod
    @property
    def MEGA(cls) -> PF:
        '''Pre-factor of `10**6`'''
        return PF( 6 )
    
    @classmethod
    @property
    def KILO(cls) -> PF:
        '''Pre-factor of `10**3`'''
        return PF( 3 )
    
    @classmethod
    @property
    def HECTO(cls) -> PF:
        '''Pre-factor of `10**2`'''
        return PF( 2 )
    
    @classmethod
    @property
    def DECA(cls) -> PF:
        '''Pre-factor of `10**1`'''
        return PF( 1 )
    
    @classmethod
    @property
    def NONE(cls) -> PF:
        '''Pre-factor of `1` or `10**0`'''
        return PF( 0 )
    
    @classmethod
    @property
    def DEZI(cls) -> PF:
        '''Pre-factor of `10**-1`'''
        return PF( -1 )
    
    @classmethod
    @property
    def CENTI(cls) -> PF:
        '''Pre-factor of `10**-2`'''
        return PF( -2 )
    
    @classmethod
    @property
    def MILLI(cls) -> PF:
        '''Pre-factor of `10**-3`'''
        return PF( -3 )
    
    @classmethod
    @property
    def MICRO(cls) -> PF:
        '''Pre-factor of `10**-6`'''
        return PF( -6 )
    
    @classmethod
    @property
    def NANO(cls) -> PF:
        '''Pre-factor of `10**-9`'''
        return PF( -9 )
    
    @classmethod
    @property
    def PICO(cls) -> PF:
        '''Pre-factor of `10**-12`'''
        return PF( -12 )
    
    @classmethod
    @property
    def FEMTO(cls) -> PF:
        '''Pre-factor of `10**-15`'''
        return PF( -15 )

    P  = PETA 
    T  = TERA 
    G  = GIGA 
    M  = MEGA 
    k  = KILO 
    h  = HECTO
    da = DECA 
    _  = NONE 
    d  = DEZI 
    c  = CENTI
    m  = MILLI
    µ  = MICRO
    n  = NANO 
    p  = PICO 
    f  = FEMTO
    
    __exp: int
    
    def __init__(self, _exp:int|PF) -> None:
        if isinstance( _exp, PF ):
            _exp = _exp.exponent
        
        self.__exp = _exp
    
    @property
    def exponent(self) -> int:
        return self.__exp
    
    @property
    def factor(self) -> int | float:
        return 10 ** self.__exp
    
    def to(self, base_si_unit:PF) -> PF:
        '''
        Use this to transform the one magnitude (`PF`) to an other magnitude (`PF`)
        
        E.g.: mg -> kg would translate to `PF.m.to(PF.k)` (this is the same as PF.m * PF.m but then its not really clear what units was it originally and what should it now be)
        '''
        return self / base_si_unit
    
    def __str__(self) -> str:
        return f"10^{self.exponent:}"
    def __repr__(self) -> str:
        return f"PF({self.exponent:})"
    
    def __mul__(self, _o:PF|Decimal|int|float|str) -> PF | Decimal:
        if isinstance( _o, (Decimal|int|float|str) ):
            return Decimal( _o ) * self.factor
        return PF( self.exponent + _o.exponent )
    __rmul__ = __mul__
    
    def __truediv__(self, _o:PF):
        if isinstance( _o, (Decimal|int|float|str) ):
            return Decimal( _o ) / self.factor
        return PF( self.exponent - _o.exponent )

    def __pow__(self, _exp:int) -> PF:
        return PF( self.exponent * _exp )

#--------------------------------------------------------------------------------------------------
# Code for developers
#--------------------------------------------------------------------------------------------------
# Logic and Expression handling #
#-------------------------------#
class Expression():
    DISABLE_SIMPLIFY: bool = False
    STR_FLOAT_ROUND : int  = 3
    
    signature: int
    '''must be unique for any Expression expect those who have the same value and the same derivatives'''
    
    def __signature__(self) -> None:
        self.signature = self.__hash__()
    
    @staticmethod
    def assert_Expression_type(*args:Sequence[Expression|Any]) -> Optional[TypeError]:
        for a in args:
            if not isinstance(a, Expression):
                raise TypeError( f"The supplied argument of type `{type(a)}` is not valid in this context, the Type must be `Expression` exclusively" )
    
    def value(self) -> Decimal: ...
    def differentiate(self, variable:Value) -> Expression: ...
    def simplify(self) -> Expression: ...
    
    def to_str( self, expression:Expression ) -> str:
        match self, expression:
            case (
                _, 
                Singleton() | Negative() | Fraction() | Pow() | Log() | Exp() | Sin() | Cos() | Tan() | Arcsin() | Arccos() | Arctan()
            ):
                return str(expression)
            
            case (
                Exp() | Sin() | Cos() | Tan() | Arcsin() | Arccos() | Arctan(),
                _
            ):
                return str(expression)
            
            case (
                _,
                Add() | Multiply() | _
            ):
                return f'({str(expression)})'

    def __str__(self) -> str: ...
    def __hash__(self) -> int: ...
    
    def __eq__(self, _o:Expression) -> bool:
        if not isinstance(_o, Expression):
           return False
        
        return self.signature == _o.signature

    def _debug(func) -> Callable[..., Expression]:
        def f(*args) -> Expression:
            self = args[0]
            
            if func.__name__ == "simplify":
                if Expression.DISABLE_SIMPLIFY:
                    return self
            
            return func(*args)
        return f

class Singleton(Expression):
    var_value: Optional[Value]
    constant : Optional[Decimal]
    
    def __init__(self, variable:Value|Decimal|int|float|str) -> None:
        assert isinstance( variable, (Value, Decimal, int, float, str) ), TypeError()
        
        if isinstance(variable, Value):
            self.var_value = variable
            self.constant  = None
        else:
            self.var_value = None
            self.constant  = Decimal( variable )

        self.__signature__()

    @cache
    def value(self) -> Decimal:
        return self.var_value._value if self.var_value else self.constant
    
    @cache
    def differentiate(self, variable: Value) -> Expression:
        return Singleton( int( math.copysign(int( self.var_value == variable ), self.var_value._value) ) if self.var_value else Decimal('0') )
    
    #@Expression._debug
    def simplify(self) -> Expression:
        if self.var_value is None:
            if self.constant < 0:
                return Negative( Singleton( -self.constant ) )
        return self
    
    @cache
    def __str__(self) -> str:
        if self.var_value:
            return self.var_value.get_id()
        
        return f'{float(self.constant):.{Expression.STR_FLOAT_ROUND}g}' # + '@S'

    def __hash__(self) -> int:
        return hash( (self.var_value, self.constant) )
class Negative(Expression):
    expression: Expression
    
    def __init__(self, expression: Expression) -> None:
        Expression.assert_Expression_type( expression )
        
        self.expression = expression.simplify()
        
        self.__signature__()
    
    @cache
    def value(self) -> Decimal:
        return -self.expression.value()
    
    @cache
    def differentiate(self, variable: Value) -> Expression:
        return Negative( self.expression.differentiate( variable ) ).simplify()

    #@Expression._debug
    def simplify(self) -> Expression:
        self.expression = self.expression.simplify()
        
        if isinstance(self.expression, Negative):
            return self.expression.expression
        
        if isinstance(self.expression, Singleton) and self.expression.constant == Decimal('0'):
            return Singleton( '0' )
        
        return self

    def to_str( self, expression:Expression ) -> str:
        match expression:
            case (
                Singleton() | Pow() | Exp() | Log() | Sin() | Cos() | Tan() | Arcsin() | Arccos() | Arctan()
            ):
                return str(expression)
            
            case (
                Add() | Multiply() | _
            ):
                return f'({str(expression)})'
    
    @cache
    def __str__(self) -> str:
        return f"-{ self.to_str(self.expression) }" # + '@N'

    def __hash__(self) -> int:
        return hash( self.expression )



class Add(Expression):
    operands: list[Expression]
    
    def __init__(self, *operands:Sequence[Expression]) -> None:
        Expression.assert_Expression_type( *operands )
        
        self.operands = [ op.simplify() for op in operands ]
        
        self.__signature__()
    
    @cache
    def value(self) -> Decimal:
        return sum( map(lambda op: op.value(), self.operands) )
    
    @cache
    def differentiate(self, variable: Value) -> Expression:
        return Add( *[op.differentiate(variable) for op in self.operands] ).simplify()

    @Expression._debug
    def simplify(self) -> Expression:
        self.operands = [ op.simplify() for op in self.operands ]
        
        # integrate children of the same type into the parent class
        other_adds: list[Add] = list( filter( lambda op: isinstance(op, Add), self.operands ) )
        
        for op in other_adds:
            self.operands.remove( op )
            self.operands.extend( op.operands )
        
        # manage all multiple operands, add(/sub) same operands
        op_counter: dict[Expression, int] = dict.fromkeys( set(map( lambda op: op.expression if isinstance(op, Negative) else op, self.operands )), 0 )
        
        for o in self.operands:
            if isinstance( o, Negative ):
                op_counter[o.expression] -= 1
            else:
                op_counter[o] += 1
        
        self.operands = [ Multiply( Singleton(str(count)), op ).simplify() for op, count in op_counter.items() ]
        
        
        # manage singles, remove zero singletons and combine constant singletons
        singles: list[Singleton] = list( filter( lambda op: isinstance(op, Singleton), self.operands ) )
        
        counter = Decimal('0')
        for op in singles:
            if op.constant is not None:
                self.operands.remove( op )
                
                counter += op.constant
        
        # all operands are constant singletons
        # we need to include this because if all operands are 0 the self.operands list would be empty and cause an exception
        if not self.operands:
            return Singleton( counter )
        
        if counter != Decimal('0'):
            self.operands.append( Singleton(counter) )
        
        
        # manage Negatives so that none is to the left
        if all( list(map( lambda op: isinstance(op, Negative), self.operands )) ):
            return Negative( Add( *[op.expression for op in self.operands] ) )
        else:
            while isinstance(self.operands[0], Negative):
                self.operands.append( self.operands[0] )
                self.operands.pop(0)
        
        if len(self.operands) == 1:
            return self.operands[0]
        
        return self
    
    @cache
    def __str__(self) -> str:
        out = str(self.operands[0])
        
        for op in self.operands[1:]:
            if isinstance(op, Negative):
                out += f' - {str(op)[1:]}'
            else:
                out += f' + {str(op)}'
                
        
        return out # + '@A'

    def __hash__(self) -> int:
        return hash( tuple(self.operands) )
class Multiply(Expression):
    operands: list[Expression]
    
    def __init__(self, *operands:Sequence[Expression]) -> None:
        Expression.assert_Expression_type( *operands )
        
        self.operands = [ op.simplify() for op in operands ]
        
        self.__signature__()
    
    @staticmethod
    def mul( *args:Sequence[object] ) -> object:
        return reduce(lambda x, y: x * y, args)
    
    @cache
    def value(self) -> Decimal:
        return Multiply.mul( *map( lambda op: op.value(), self.operands ) )
    
    @cache
    def differentiate(self, variable: Value) -> Expression:
        return Add( *[Multiply( *[op.differentiate(variable), *(self.operands[:i] + self.operands[i+1:])] ) for i, op in enumerate(self.operands)] ).simplify()

    @Expression._debug
    def simplify(self) -> Expression:
        self.operands = [ op.simplify() for op in self.operands ]
        
        # integrate children of the same type into the parent class
        other_multipliers: list[Multiply] = list( filter( lambda op: isinstance(op, Multiply), self.operands ) )
        
        for op in other_multipliers:
            self.operands.extend( op.operands )
            self.operands.remove( op )
        
        
        # manage Negatives NoOp for count(Negatives)%2==0 else Negative(abs(operands))
        negatives: list[Negative] = list(filter( lambda op: isinstance(op, Negative), self.operands ))
        
        for n in negatives:
            self.operands.remove( n )
            self.operands.append( n.expression )
        
        if len(negatives) % 2 == 1:
            return Negative( self ).simplify()
        
        
        # manage singles, remove zero singletons and combine constant singletons
        singles: list[Singleton] = list( filter( lambda op: isinstance(op, Singleton), self.operands ) )
        
        if any( map( lambda s: s.constant == Decimal('0'), singles ) ):
            return Singleton( '0' )
        
        counter = Decimal('1')
        for s in singles:
            if s.constant is not None:
                self.operands.remove( s )
                
                counter *= s.constant
        
        # all operands are constant singletons
        # we need to include this because if all operands are 1 the self.operands list would be empty and cause an exception
        if not self.operands:
            return Singleton( counter )
        
        if counter != Decimal('1'):
            self.operands.append( Singleton(counter) )
        
        # manage and combine Fractions
        fractions: list[Fraction] = list( filter( lambda op: isinstance(op, Fraction), self.operands ) )
        
        for frac in fractions:
            self.operands.remove(frac)
        
        if fractions:
            self.operands.append(
                Fraction( 
                         Multiply( *[f.numerator for f in fractions] ),
                         Multiply( *[f.denominator for f in fractions] )
                         ).simplify()
            )
        
        if len(self.operands) == 1:
            return self.operands[0]
        
        return self
    
    @cache
    def __str__(self) -> str:
        return ' * '.join( map( lambda op: self.to_str(op), self.operands ) ) # + '@M'

    def __hash__(self) -> int:
        return hash( tuple(self.operands) )
class Fraction(Expression):
    numerator  : Expression
    denominator: Expression
    
    def __init__(self, numerator:Expression, denominator:Expression) -> None:
        Expression.assert_Expression_type( numerator, denominator )
        assert denominator.value() != 0, ValueError( "The denominator can not the 0" )
        
        self.numerator   = numerator.simplify()
        self.denominator = denominator.simplify()
        
        self.__signature__()
    
    @cache
    def value(self) -> Decimal:
        return self.numerator.value() / self.denominator.value()
    
    @cache
    def differentiate(self, variable: Value) -> Expression:
        # d/dx numerator(x)/denominator(x) = (denominator(x) * (d/dx numerator(x)) - numerator(x) * (d/dx denominator(x)))/denominator(x)**2
        return Fraction( 
                        Add( 
                            Multiply( 
                                     self.denominator,
                                     self.numerator.differentiate(variable)
                                     ),
                            Negative( 
                                     Multiply( 
                                              self.denominator.differentiate(variable),
                                              self.numerator
                                              )
                                     )
                            ),
                        Pow( 
                            self.denominator,
                            Singleton( '2')
                            )
                        ).simplify()

    @Expression._debug
    def simplify(self) -> Expression:
        self.numerator   = self.numerator.simplify()
        self.denominator = self.denominator.simplify()
        
        if self.numerator == self.denominator:
            return Singleton( '1' )
        
        if isinstance(self.denominator, Singleton) and self.denominator.constant == Decimal('1'):
            return self.numerator
        
        if isinstance(self.numerator, Fraction):
            return Fraction( 
                            self.numerator.numerator, 
                            Multiply( 
                                     self.numerator.denominator,
                                     self.denominator
                                     ) 
                            ).simplify()
        
        if isinstance(self.numerator, Singleton) and self.numerator.constant == Decimal('0'):
            return Singleton( '0' )
        
        if isinstance(self.denominator, Fraction):
            return Fraction( 
                            Multiply( 
                                     self.numerator,
                                     self.denominator.denominator
                                     ), 
                            self.denominator.numerator
                            ).simplify()
        
        if isinstance(self.denominator, Negative):
            return Fraction( 
                            Negative( 
                                     self.numerator
                                     ),
                            self.denominator.expression
                            ).simplify()
        
        if isinstance(self.denominator, Pow): 
            if isinstance( self.denominator.exponent, Negative ):
                return Multiply( 
                                self.numerator,
                                Pow( 
                                    self.denominator.base,
                                    self.denominator.exponent.expression
                                    ) 
                                ).simplify()
            
            if isinstance( self.denominator.exponent, Singleton ) and self.denominator.exponent.constant and self.denominator.exponent.constant < Decimal('0'):
                return Multiply( 
                                self.numerator,
                                Pow( 
                                    self.denominator.base,
                                    Singleton( -self.denominator.exponent.constant )
                                    ) 
                                ).simplify()

            if isinstance(self.numerator, Singleton) and self.numerator.constant and isinstance(self.denominator, Singleton) and self.denominator.constant:
                return Singleton( self.value() )
            
        return self
    
    
    def to_str( self, expression:Expression ) -> tuple[str, str]:
        match self.numerator:
            case (
                Singleton() | Negative() | Pow() | Exp() | Log() | Sin() | Cos() | Tan() | Arcsin() | Arccos() | Arctan()
            ):
                num = str(self.numerator)
            
            case (
                Fraction() | Add() | Multiply() | _
            ):
                num = f'({str(self.numerator)})'
        
        match self.denominator:
            case (
                Singleton() | Pow() | Exp() | Log() | Sin() | Cos() | Tan() | Arcsin() | Arccos() | Arctan()
            ):
                denom = str(self.denominator)
            
            case (
                Negative() | Fraction() | Add() | Multiply() | _
            ):
                denom = f'({str(self.denominator)})'
        
        return num, denom
    
    @cache
    def __str__(self) -> str:
        return '{:s}/{:s}'.format( *self.to_str(None) )
    
    def __hash__(self) -> int:
        return hash( (self.numerator, self.denominator) )


class Pow(Expression):
    base    : Expression
    exponent: Expression
    
    def __init__(self, base: Expression, exponent: Expression) -> None:
        Expression.assert_Expression_type( base, exponent )
        
        base     = base.simplify()
        exponent = exponent.simplify()
        
        assert base.value() >= 0 or is_integer( exponent.value() ), ValueError( "the value of the base can not be negative for non integer exponent values" )
        assert base.value() != 0 or exponent.value() != 0, ValueError( "the base and the exponent can not both be 0" )
        
        self.base     = base
        self.exponent = exponent
        
        self.__signature__()
    
    @cache
    def value(self) -> Decimal:
        return self.base.value() ** self.exponent.value()
    
    @cache
    def differentiate(self, variable: Value) -> Expression:
        # d/dx (base(x)**exponent(x)) = base(x)**(exponent(x) - 1) * ( exponent(x) * (d/dx base(x)) + (d/dx exponent(x)) * base(x) * ln( base(x) ) )
        return Multiply( 
                        Pow( 
                            self.base,
                            Add( 
                                self.exponent,
                                Singleton('-1')
                            )
                        ),
                        Add( 
                            Multiply( 
                                     self.exponent,
                                     self.base.differentiate(variable)
                                     ),
                            Multiply( 
                                     self.exponent.differentiate(variable),
                                     self.base,
                                     Log( 
                                         self.base
                                         )
                                     )
                            )
                        ).simplify()

    @Expression._debug
    def simplify(self) -> Expression:
        self.base     = self.base.simplify()
        self.exponent = self.exponent.simplify()
        
        if isinstance(self.base, Singleton) and self.base.constant == Decimal('0'):
            return Singleton( '0' )
        
        if isinstance(self.exponent, Singleton) and self.exponent.constant:
            if self.exponent.constant == Decimal('0'):
                return Singleton( '1' )
            elif self.exponent.constant == Decimal('1'):
                return self.base
        
        if  isinstance(self.base, Singleton) and self.base.constant \
        and isinstance(self.exponent, Singleton) and self.exponent.constant:
            return Singleton( self.value() )
        
        return self
    
    def to_str( self, expression:Expression ) -> tuple[str, str]:
        match self.base:
            case (
                Singleton()
            ):
                b = str(self.base)
            
            case (
                Negative() | Exp() | Log() | Sin() | Cos() | Tan() | Arcsin() | Arccos() | Arctan() | Add() | Multiply() | Pow() | _
            ):
                b = f'({str(self.base)})'
        
        match self.exponent:
            case (
                Singleton() | Exp() | Log() | Sin() | Cos() | Tan() | Arcsin() | Arccos() | Arctan()
            ):
                e = str(self.exponent)
            
            case (
                Add() | Multiply() | Pow() | _
            ):
                e =  f'({str(self.exponent)})'
        
        return b, e
    
    @cache
    def __str__(self) -> str:
        return '{:s}**{:s}'.format( *self.to_str(None) )

    def __hash__(self) -> int:
        return hash( (self.base, self.exponent) )
class Exp(Expression):
    exponent: Expression
    
    def __init__(self, exponent: Expression) -> None:
        Expression.assert_Expression_type( exponent )
        
        self.exponent = exponent.simplify()
        
        self.__signature__()
    
    @cache
    def value(self) -> Decimal:
        return self.exponent.value().exp()
    
    @cache
    def differentiate(self, variable: Value) -> Expression:
        return Multiply( 
                        self,
                        self.exponent.differentiate(variable)
                        ).simplify()

    @Expression._debug
    def simplify(self) -> Expression:
        self.exponent = self.exponent.simplify()
        
        if isinstance(self.exponent, Log) and not self.exponent.base:
            return self.exponent
        
        return self

    @cache
    def __str__(self) -> str:
        return f'exp({self.to_str(self.exponent)})'
    
    def __hash__(self) -> int:
        return hash( self.exponent )
class Log(Expression):
    base    : Optional[Expression]
    argument: Expression
    
    def __init__(self, argument:Expression, base:Optional[Expression]=None) -> None:
        Expression.assert_Expression_type( argument )
        
        argument = argument.simplify()
        if base:
            Expression.assert_Expression_type( base )
            base = base.simplify()
        
        assert base is None or base.value() > 0, ValueError( "The supplied base must be positive or None defaulting to the natural logarithm" )
        assert argument.value() > 0, ValueError( "The supplied argument must be positive" )
        
        self.base     = base
        self.argument = argument
        
        self.__signature__()
    
    @cache
    def value(self) -> Decimal:
        if not self.base:
            return self.argument.value().ln()
        
        return self.argument.value().ln() / self.base.value().ln()
    
    @cache
    def differentiate(self, variable: Value) -> Expression:
        # L(x) = d/dx (ln(argument(x))/ln(base(x))) 
        #      = d/dx log_{base(x)}(argument(x)) = ( (d/dx argument(x)) / argument(x) - L(x) * (d/dx base(x)) / base(x) ) / ln(base(x))
        # here: log(...) is the natural logarithm
        if not self.base:
            return Fraction(
                self.argument.differentiate( variable ),
                self.argument
            ).simplify()
        
        return Fraction( 
                        Add( 
                            Fraction( 
                                     self.argument.differentiate(variable), 
                                     self.argument
                                     ), 
                            Negative( 
                                     Fraction( 
                                              Multiply( 
                                                       self, 
                                                       self.base.differentiate(variable)
                                                       ),
                                              self.base
                                              )
                                     )
                            ), 
                        Log( self.base )
                        ).simplify()

    @Expression._debug
    def simplify(self) -> Expression:
        self.base     = self.base if self.base is None else self.base.simplify()
        self.argument = self.argument.simplify()
        
        if self.base and self.argument == self.base:
            return Singleton( '1' )
        
        if isinstance(self.argument, Pow):
            return Multiply( 
                            self.argument.exponent,
                            Log( self.argument.base, self.base )
                            ).simplify()
        
        if isinstance(self.argument, Singleton) and self.argument.constant == Decimal('1'):
            return Singleton( '0' )
        
        return self
    

    def to_str( self, expression:Expression ) -> str:
        match self.base:
            case None:
                return 'None'
                
            case (
                Singleton()
            ):
                return str(self.base)
            
            case (
                Negative() | Exp() | Log() | Sin() | Cos() | Tan() | Arcsin() | Arccos() | Arctan() | Add() | Multiply() | Pow() | _
            ):
                return f'({str(self.base)})'
    
    @cache
    def __str__(self) -> str:
        return (f'log_{self.to_str(None)}' if self.base else 'ln') + f'({str(self.argument)})'

    def __hash__(self) -> int:
        return hash( (self.base, self.argument) )


class Sin(Expression): 
    expression: Expression
    
    def __init__(self, expression: Expression) -> None:
        Expression.assert_Expression_type( expression )
        
        self.expression = expression.simplify()

        self.__signature__()
    
    @cache
    def value(self) -> Decimal:
        return Decimal( math.sin( float( self.expression.value() ) ) )
    
    @cache
    def differentiate(self, variable: Value) -> Expression:
        return Multiply( 
                        Cos( self.expression ),
                        self.expression.differentiate(variable)
                        ).simplify()

    @Expression._debug
    def simplify(self) -> Expression:
        self.expression = self.expression.simplify()
        
        if isinstance(self.expression, Arcsin):
            return self.expression.expression
        
        return self
    
    @cache
    def __str__(self) -> str:
        return f"sin({self.expression})"

    def __hash__(self) -> int:
        return hash( self.expression )
class Cos(Expression):
    expression: Expression
    
    def __init__(self, expression: Expression) -> None:
        Expression.assert_Expression_type( expression )

        self.expression = expression.simplify()

        self.__signature__()
    
    @cache
    def value(self) -> Decimal:
        return Decimal( math.cos( float( self.expression.value() ) ) )
    
    @cache
    def differentiate(self, variable: Value) -> Expression:
        return Negative( 
                        Multiply( 
                                 Sin( self.expression ),
                                 self.expression.differentiate(variable)
                                 ) 
                        ).simplify()

    @Expression._debug
    def simplify(self) -> Expression:
        self.expression = self.expression.simplify()
        
        if isinstance(self.expression, Arccos):
            return self.expression.expression
        
        return self
    
    @cache
    def __str__(self) -> str:
        return f"cos({self.expression})"

    def __hash__(self) -> int:
        return hash( self.expression )
class Tan(Expression): 
    expression: Expression
    
    def __init__(self, expression: Expression) -> None:
        Expression.assert_Expression_type( expression )
        
        assert not is_integer( expression.value() / Decimal( math.pi ) - Decimal('0.5') ), ValueError( "Expression must not evaluate to a multiple of `expression`/pi - 0.5 " )

        self.expression = expression.simplify()

        self.__signature__()
    
    @cache
    def value(self) -> Decimal:
        return Decimal( math.tan( float( self.expression.value() ) ) )
    
    @cache
    def differentiate(self, variable: Value) -> Expression:
        return Fraction( 
                        self.expression.differentiate(variable),
                        Pow( 
                            Cos( self.expression ),
                            Singleton( '2' )
                            )
                        ).simplify()

    @Expression._debug
    def simplify(self) -> Expression:
        self.expression = self.expression.simplify()
        
        if isinstance(self.expression, Arctan):
            return self.expression.expression
        
        return self
    
    @cache
    def __str__(self) -> str:
        return f"tan({self.expression})"

    def __hash__(self) -> int:
        return hash( self.expression )


class Arcsin(Expression):
    expression: Expression
    
    def __init__(self, expression: Expression) -> None:
        Expression.assert_Expression_type( expression )

        assert -1 <= expression.value() <= 1, ValueError( "Expression must be in region [-1, 1]" )

        self.expression = expression.simplify()

        self.__signature__()
    
    @cache
    def value(self) -> Decimal:
        return Decimal( math.asin( float( self.expression.value() ) ) )
    
    @cache
    def differentiate(self, variable: Value) -> Expression:
        # d/dx asin(f(x)) = (d/dx f(x)) / sqrt(1 - f(x)**2)
        return Fraction( 
                        self.expression.differentiate(variable),
                        Pow( 
                            Add( 
                                Singleton('1'),
                                Negative( 
                                         Pow( 
                                             self.expression,
                                             Singleton('2')
                                             )
                                         )
                                ),
                            Singleton('0.5')
                            )
                        ).simplify()

    @Expression._debug
    def simplify(self) -> Expression:
        self.expression = self.expression.simplify()
        
        if isinstance(self.expression, Sin):
            return self.expression.expression
        
        return self
    
    @cache
    def __str__(self) -> str:
        return f"asin({self.expression})"

    def __hash__(self) -> int:
        return hash( self.expression )
class Arccos(Expression):
    expression: Expression
    
    def __init__(self, expression: Expression) -> None:
        Expression.assert_Expression_type( expression )

        assert -1 <= expression.value() <= 1, ValueError( "Expression must be in region [-1, 1]" )

        self.expression = expression.simplify()

        self.__signature__()
    
    @cache
    def value(self) -> Decimal:
        return Decimal( math.acos( float( self.expression.value() ) ) )
    
    @cache
    def differentiate(self, variable: Value) -> Expression:
        # d/dx acos(f(x)) = -d/dx asin(f(x)) = -(d/dx f(x)) / sqrt(1 - f(x)**2)
        return Negative( Arcsin(self.expression).differentiate(variable) ).simplify()

    @Expression._debug
    def simplify(self) -> Expression:
        self.expression = self.expression.simplify()
        
        if isinstance(self.expression, Cos):
            return self.expression.expression
        
        return self
    
    @cache
    def __str__(self) -> str:
        return f"acos({self.expression})"

    def __hash__(self) -> int:
        return hash( self.expression )
class Arctan(Expression):
    expression: Expression
    
    def __init__(self, expression: Expression) -> None:
        Expression.assert_Expression_type( expression )

        self.expression = expression.simplify()

        self.__signature__()
    
    @cache
    def value(self) -> Decimal:
        return Decimal( math.atan( float( self.expression.value() ) ) )
    
    @cache
    def differentiate(self, variable: Value) -> Expression:
        # d/dx atan(f(x)) = (d/dx f(x)) / (f(x)**2 + 1)
        return Fraction( 
                        self.expression.differentiate(variable), 
                        Add( 
                            Pow( 
                                self.expression,
                                Singleton('2')
                                ), 
                            Singleton('1') 
                            )
                        ).simplify()

    @Expression._debug
    def simplify(self) -> Expression:
        self.expression = self.expression.simplify()
        
        if isinstance(self.expression, Tan):
            return self.expression.expression
        
        return self
    
    @cache
    def __str__(self) -> str:
        return f"atan({self.expression})"

    def __hash__(self) -> int:
        return hash( self.expression )


class Sinh(Expression): 
    expression: Expression
    
    def __init__(self, expression: Expression) -> None:
        Expression.assert_Expression_type( expression )
        
        self.expression = expression.simplify()

        self.__signature__()
    
    @cache
    def value(self) -> Decimal:
        return Multiply( 
                        Singleton( '0.5' ), 
                        Add( 
                            Exp( self.expression ),
                            Negative( Exp( Negative( self.expression ) ) )
                            )
                        ).simplify().value()
    
    @cache
    def differentiate(self, variable: Value) -> Expression:
        return Multiply( 
                        Cosh( self.expression ),
                        self.expression.differentiate(variable)
                        ).simplify()

    @Expression._debug
    def simplify(self) -> Expression:
        self.expression = self.expression.simplify()
        
        if isinstance(self.expression, Arsinh):
            return self.expression.expression
        
        return self
    
    @cache
    def __str__(self) -> str:
        return f"sinh({self.expression})"

    def __hash__(self) -> int:
        return hash( self.expression )
class Cosh(Expression): 
    expression: Expression
    
    def __init__(self, expression: Expression) -> None:
        Expression.assert_Expression_type( expression )

        self.expression = expression.simplify()

        self.__signature__()
    
    @cache
    def value(self) -> Decimal:
        return Multiply( 
                        Singleton( '0.5' ), 
                        Add( 
                            Exp( self.expression ),
                            Exp( Negative( self.expression ) ) 
                            )
                        ).simplify().value()
    
    @cache
    def differentiate(self, variable: Value) -> Expression:
        return Multiply( 
                        Sinh( self.expression ),
                        self.expression.differentiate(variable)
                        ).simplify()

    @Expression._debug
    def simplify(self) -> Expression:
        self.expression = self.expression.simplify()
        
        if isinstance(self.expression, Arcosh):
            return self.expression.expression
        
        return self
    
    @cache
    def __str__(self) -> str:
        return f"cos({self.expression})"

    def __hash__(self) -> int:
        return hash( self.expression )
class Tanh(Expression): 
    expression: Expression
    
    def __init__(self, expression: Expression) -> None:
        Expression.assert_Expression_type( expression )

        self.expression = expression.simplify()

        self.__signature__()
    
    @cache
    def value(self) -> Decimal:
        return Add( 
                   Singleton( '1' ),
                   Negative( 
                            Fraction( 
                                     Singleton( '2' ),
                                     Add( 
                                         Exp( 
                                             Multiply( 
                                                      Singleton( '2' ),
                                                      self.expression
                                                      ) 
                                             ),
                                         Singleton( '1' )
                                         )
                                     )
                            )
                   )
    
    @cache
    def differentiate(self, variable: Value) -> Expression:
        return Fraction( 
                        self.expression.differentiate(variable),
                        Pow( 
                            Cosh( self.expression ),
                            Singleton( '2' )
                            )
                        ).simplify()

    @Expression._debug
    def simplify(self) -> Expression:
        self.expression = self.expression.simplify()
        
        if isinstance(self.expression, Artanh):
            return self.expression.expression
        
        return self
    
    @cache
    def __str__(self) -> str:
        return f"tan({self.expression})"

    def __hash__(self) -> int:
        return hash( self.expression )


class Arsinh(Expression):
    expression: Expression
    
    def __init__(self, expression: Expression) -> None:
        Expression.assert_Expression_type( expression )

        self.expression = expression.simplify()

        self.__signature__()
    
    @cache
    def value(self) -> Decimal:
        return Log( 
                   Add( 
                       self.expression,
                       Pow( 
                           Add( 
                               Pow( 
                                   self.expression,
                                   Singleton('2')
                                   ),
                               Singleton('1')
                               ),
                           Singleton('0.5')
                           )
                       )
                   ).simplify().value()
    
    @cache
    def differentiate(self, variable: Value) -> Expression:
        # d/dx arsinh(f(x)) = (d/dx f(x)) / sqrt(1 + f(x)**2)
        return Fraction( 
                        self.expression.differentiate(variable),
                        Pow( 
                            Add( 
                                Singleton('1'),
                                Pow( 
                                    self.expression,
                                    Singleton('2')
                                    )
                                ),
                            Singleton('0.5')
                            )
                        ).simplify()

    @Expression._debug
    def simplify(self) -> Expression:
        self.expression = self.expression.simplify()
        
        if isinstance(self.expression, Sinh):
            return self.expression.expression
        
        return self
    
    @cache
    def __str__(self) -> str:
        return f"arsinh({self.expression})"

    def __hash__(self) -> int:
        return hash( self.expression )
class Arcosh(Expression):
    expression: Expression
    
    def __init__(self, expression: Expression) -> None:
        Expression.assert_Expression_type( expression )
        
        assert 1 <= expression.value(), ValueError( "Expression must be in region [1, +inf)" )
        
        self.expression = expression.simplify()

        self.__signature__()
    
    @cache
    def value(self) -> Decimal:
        return Log( 
                   Add( 
                       self.expression,
                       Pow( 
                           Add( 
                               Pow( 
                                   self.expression,
                                   Singleton('2')
                                   ),
                               Singleton('-1')
                               ),
                           Singleton('0.5')
                           )
                       )
                   ).simplify().value()
    
    @cache
    def differentiate(self, variable: Value) -> Expression:
        # d/dx arcosh(f(x)) = (d/dx f(x)) / sqrt(f(x)**2 - 1)
        return Fraction( 
                        self.expression.differentiate(variable),
                        Pow( 
                            Add( 
                                Pow( 
                                    self.expression,
                                    Singleton('2')
                                    ),
                                Negative( Singleton('1') )
                                ),
                            Singleton('0.5')
                            )
                        ).simplify()

    @Expression._debug
    def simplify(self) -> Expression:
        self.expression = self.expression.simplify()
        
        if isinstance(self.expression, Cosh):
            return self.expression.expression
        
        return self
    
    @cache
    def __str__(self) -> str:
        return f"arcosh({self.expression})"

    def __hash__(self) -> int:
        return hash( self.expression )
class Artanh(Expression):
    expression: Expression
    
    def __init__(self, expression: Expression) -> None:
        Expression.assert_Expression_type( expression )

        assert -1 <= expression.value() <= 1, ValueError( "Expression must be in region [-1, 1]" )
        
        self.expression = expression.simplify()

        self.__signature__()
    
    @cache
    def value(self) -> Decimal:
        return Multiply( 
                        Singleton('0.5'),
                        Log( 
                            Fraction( 
                                     Add( 
                                         Singleton('1'),
                                         self.expression
                                         ),
                                     Add( 
                                         Singleton('1'),
                                         Negative( self.expression )
                                         )
                                     )
                            )
                        ).simplify().value()
    
    @cache
    def differentiate(self, variable: Value) -> Expression:
        # d/dx artanh(f(x)) = (d/dx f(x)) / (1 - f(x)**2)
        return Fraction( 
                        self.expression.differentiate(variable), 
                        Add( 
                            Singleton('1'),
                            Negative( 
                                     Pow( 
                                         self.expression,
                                         Singleton('2')
                                         )
                                     )
                            )
                        ).simplify()

    @Expression._debug
    def simplify(self) -> Expression:
        self.expression = self.expression.simplify()
        
        if isinstance(self.expression, Tanh):
            return self.expression.expression
        
        return self
    
    @cache
    def __str__(self) -> str:
        return f"artanh({self.expression})"

    def __hash__(self) -> int:
        return hash( self.expression )


def _type_error_input(_x:Any) -> TypeError:
    return TypeError( f"The supplied value of type `{type(_x)}` is not valid in this context, the Type must be one of `Value`, `Decimal`, `int`, `float`, `str`" )
def _type_error_input_kwargs(**_kwargs:dict[Any]) -> TypeError:
    return TypeError( f"The supplied values of types { ','.join( map( lambda item: f'{str(item[0])}: `{str(item[1])}`', _kwargs.items() ) ) } are not valid in this context, the Types must be one of `Value`, `Decimal`, `int`, `float`, `str`" )


#---------------#
# End User Code #
#---------------#
# ---------------#
# Math functions #
# ---------------#
def exp( _x:Value|Decimal|int|float|str, / ) -> Value:
    match _x:
        case Value():
            vars, _x_expression = _x._operation_setup( _x )

            return Value._init_by_expression(
                    Exp( _x_expression ),
                    vars
                )
        
        case Decimal() | int() | float() | str():
            return Value._init_by_expression(
                    Exp( Singleton(_x) ),
                    {}
                )
        
        case _:
            raise _type_error_input(_x)
def log( argument:Value|Decimal|int|float|str, base:Optional[Value|Decimal|int|float|str]=None ) -> Value:
    match (argument, base):
        case ( Value(), Value() ):
            vars_argument, argument_expression = argument._operation_setup( argument )
            vars_base    , base_expression     = base._operation_setup( base )

            return Value._init_by_expression(
                Log( argument_expression, base_expression ),
                vars_argument.union( vars_base )
            )
            
        case ( Value(), None | Decimal() | int() | float() | str() ):
            vars_argument, argument_expression = argument._operation_setup( argument )

            return Value._init_by_expression(
                Log( argument_expression, Singleton(base) if base else None ),
                vars_argument
            )
            
        case ( Decimal() | int() | float() | str(), Value() ):
            vars_base, base_expression = base._operation_setup( base )
            
            return Value._init_by_expression(
                Log( Singleton( argument ), base_expression ),
                vars_base
            )
    
        case ( Decimal() | int() | float() | str(), None | Decimal() | int() | float() | str() ):
            return Value._init_by_expression(
                Log( Singleton( argument ), Singleton( base ) if base else None ),
                {}
            )
        
        case (_, _):
            raise _type_error_input_kwargs( argument=argument, base=base )


def sin( _x:Value|Decimal|int|float|str, / ) -> Value:
    match _x:
        case Value():
            vars, _x_expression = _x._operation_setup( _x )

            return Value._init_by_expression(
                    Sin( _x_expression ),
                    vars
                )
        
        case Decimal() | int() | float() | str():
            return Value._init_by_expression(
                    Sin( Singleton(_x) ),
                    {}
                )
        
        case _:
            raise _type_error_input(_x)
def cos( _x:Value|Decimal|int|float|str, / ) -> Value:
    match _x:
        case Value():
            vars, _x_expression = _x._operation_setup( _x )

            return Value._init_by_expression(
                    Cos( _x_expression ),
                    vars
                )
        
        case Decimal() | int() | float() | str():
            return Value._init_by_expression(
                    Cos( Singleton(_x) ),
                    {}
                )
        
        case _:
            raise _type_error_input(_x)
def tan( _x:Value|Decimal|int|float|str, / ) -> Value:
    match _x:
        case Value():
            vars, _x_expression = _x._operation_setup( _x )

            return Value._init_by_expression(
                    Tan( _x_expression ),
                    vars
                )
        
        case Decimal() | int() | float() | str():
            return Value._init_by_expression(
                    Tan( Singleton(_x) ),
                    {}
                )
        
        case _:
            raise _type_error_input(_x)


def arcsin( _x:Value|Decimal|int|float|str, / ) -> Value:
    match _x:
        case Value():
            vars, _x_expression = _x._operation_setup( _x )

            return Value._init_by_expression(
                    Arcsin( _x_expression ),
                    vars
                )
        
        case Decimal() | int() | float() | str():
            return Value._init_by_expression(
                    Arcsin( Singleton(_x) ),
                    {}
                )
        
        case _:
            raise _type_error_input(_x)
def arccos( _x:Value|Decimal|int|float|str, / ) -> Value:
    match _x:
        case Value():
            vars, _x_expression = _x._operation_setup( _x )

            return Value._init_by_expression(
                    Arccos( _x_expression ),
                    vars
                )
        
        case Decimal() | int() | float() | str():
            return Value._init_by_expression(
                    Arccos( Singleton(_x) ),
                    {}
                )
        
        case _:
            raise _type_error_input(_x)
def arctan( _x:Value|Decimal|int|float|str, / ) -> Value:
    match _x:
        case Value():
            vars, _x_expression = _x._operation_setup( _x )

            return Value._init_by_expression(
                    Arctan( _x_expression ),
                    vars
                )
        
        case Decimal() | int() | float() | str():
            return Value._init_by_expression(
                    Arctan( Singleton(_x) ),
                    {}
                )
        
        case _:
            raise _type_error_input(_x)


def sinh( _x:Value|Decimal|int|float|str, / ) -> Value:
    match _x:
        case Value():
            vars, _x_expression = _x._operation_setup( _x )

            return Value._init_by_expression(
                    Sinh( _x_expression ),
                    vars
                )
        
        case Decimal() | int() | float() | str():
            return Value._init_by_expression(
                    Sinh( Singleton(_x) ),
                    {}
                )
        
        case _:
            raise _type_error_input(_x)
def cosh( _x:Value|Decimal|int|float|str, / ) -> Value:
    match _x:
        case Value():
            vars, _x_expression = _x._operation_setup( _x )

            return Value._init_by_expression(
                    Cosh( _x_expression ),
                    vars
                )
        
        case Decimal() | int() | float() | str():
            return Value._init_by_expression(
                    Cosh( Singleton(_x) ),
                    {}
                )
        
        case _:
            raise _type_error_input(_x)
def tanh( _x:Value|Decimal|int|float|str, / ) -> Value:
    match _x:
        case Value():
            vars, _x_expression = _x._operation_setup( _x )

            return Value._init_by_expression(
                    Tanh( _x_expression ),
                    vars
                )
        
        case Decimal() | int() | float() | str():
            return Value._init_by_expression(
                    Tanh( Singleton(_x) ),
                    {}
                )
        
        case _:
            raise _type_error_input(_x)


def arsinh( _x:Value|Decimal|int|float|str, / ) -> Value:
    match _x:
        case Value():
            vars, _x_expression = _x._operation_setup( _x )

            return Value._init_by_expression(
                    Arsinh( _x_expression ),
                    vars
                )
        
        case Decimal() | int() | float() | str():
            return Value._init_by_expression(
                    Arsinh( Singleton(_x) ),
                    {}
                )
        
        case _:
            raise _type_error_input(_x)
def arcosh( _x:Value|Decimal|int|float|str, / ) -> Value:
    match _x:
        case Value():
            vars, _x_expression = _x._operation_setup( _x )

            return Value._init_by_expression(
                    Arcosh( _x_expression ),
                    vars
                )
        
        case Decimal() | int() | float() | str():
            return Value._init_by_expression(
                    Arcosh( Singleton(_x) ),
                    {}
                )
        
        case _:
            raise _type_error_input(_x)
def artanh( _x:Value|Decimal|int|float|str, / ) -> Value:
    match _x:
        case Value():
            vars, _x_expression = _x._operation_setup( _x )

            return Value._init_by_expression(
                    Artanh( _x_expression ),
                    vars
                )
        
        case Decimal() | int() | float() | str():
            return Value._init_by_expression(
                    Artanh( Singleton(_x) ),
                    {}
                )
        
        case _:
            raise _type_error_input(_x)


class Value():
    PRINT_MODE      : Print_mode        = Print_mode.DEFAULT
    PREC_MODE       : Precision_mode    = Precision_mode.DEFAULT
    TRIG_MODE       : Trigonometry_mode = Trigonometry_mode.DEFAULT
    PROPAGATION_MODE: Propagation_mode  = Propagation_mode.DEFAULT
    
    def __init__(self, value:str|int|float|Decimal, error:str|int|float|Decimal, exp:int|PF=0, prec:int=2, *, id:Optional[str]=None) -> None:
        """
        Args:
            value (`str | int | float | Decimal`): value (e.g. of measurement)
            error (`str | int | float | Decimal`): error (e.g. of measurement)
            exp (`int` | `PF`, optional): shift exponent of the given value and error, same as multiplying the given value and error by 10**exp. Defaults to 0.
            prec (`int`, optional): amount of significant digits used for representation. Defaults to 2.
            id (`Optional[str]`): only used in `print_info_equation()`. If supplied the id string is being inserted into the equations of `print_info_equation()` otherwise if None every instance of this `Value` will be rendered by calling `repr`
        """
        assert isinstance( value, ( str, int, float, Decimal ) ), TypeError ("value must be of type str or int or float or Decimal")
        assert isinstance( error, ( str, int, float, Decimal ) ), TypeError ("error must be of type str or int or float or Decimal")
        assert isinstance( exp,   ( int, PF ) ),                  TypeError ("exp must be of type int")
        assert isinstance( prec, int ),                           TypeError ("prec must be of type int")
        assert prec >= 0,                                         ValueError("prec must be non negative")
        
        self.__value = Value.shift_separator_left( Decimal(value), PF( exp ).exponent )
        self.__error = Value.shift_separator_left( Decimal(error), PF( exp ).exponent )
        
        assert self.__error >= 0, ValueError("error must be non negative")
        
        self.__exp  = self.__error.adjusted()+1
        self.__prec = prec
        
        self.__value = Value.shift_separator_left( self.__value, -self.__exp )
        self.__error = Value.shift_separator_left( self.__error, -self.__exp )
        
        
        self.__id          = id.strip() if id else None
        self._variables    = { self }
        self._expression   = Singleton( self )
        self._is_primitive = True
    
    @property
    def value(self) -> Decimal:
        return self.__round_shift( self.__value )
    
    @property
    def error(self) -> Decimal:
        return self.__round_shift( self.__error )
    
    @property
    def relative_error(self) -> Decimal:
        rel_exp=self._relative_error.adjusted()+1
        return self.__round_shift( Value.shift_separator_left( self._relative_error, -rel_exp ), rel_exp )
    
    @property
    def v(self) -> Decimal:
        '''alias for `value`'''
        return self.value
    @property
    def e(self) -> Decimal:
        '''alias for `error`'''
        return self.error
    @property
    def re(self) -> Decimal:
        '''alias for `relative_error`'''
        return self.relative_error
    
    def str_relative_error(self) -> str:
        return f"δe= {self.relative_error}"
    
    def print_info_equation(self, round_decimals_to:Optional[int]=3) -> None:
        '''print the underlying equation of this `Value` instance and the associated derivatives'''
        r = Expression.STR_FLOAT_ROUND
        Expression.STR_FLOAT_ROUND = round_decimals_to
        
        print( "Equation   : q = ", str(self._expression) )
        print( "Derivatives:" )
        print( *[f"\t∂q/∂{var.get_id()}:\t{str(self._expression.differentiate(var))}" for var in self._variables], sep='\n', end='\n\n' )

        Expression.STR_FLOAT_ROUND = r
    
    @staticmethod
    def to_Value(_o:Value|Decimal|int|float|str, /) -> Value:
        match _o:
            case Value():
                return _o
            case Decimal() | int() | float() | str():
                return Value( _o, '0' )
            case _:
                raise TypeError()
    
    @classmethod
    def weighted_average( cls, *values:Sequence[Value] ) -> Value:
        assert all( map( lambda v: isinstance(v, Value), values ) ), TypeError("All arguments must be of type Value")
        
        prec_list = map(lambda var: var.__prec, values)
        prec      = max( prec_list ) if Value.PREC_MODE == Precision_mode.TOWARDS_MAX else min( prec_list )
        
        sum_weights = sum( map( lambda v: v.error**Decimal('-2'), values ) ) # weights are in this case same as 1/(value.error)**2
        
        return Value(
            sum(map( lambda v: v.value * v.error**Decimal('-2'), values )) / sum_weights,
            sum_weights ** Decimal('-0.5'),
            prec=prec
        )
    
    #----------------------------------------------------------------------------------------------
    # private calculation stuff
    #----------------------------------------------------------------------------------------------
    __id: Optional[str]
    
    __prec: int
    
    __value: Decimal
    __error: Decimal
    __exp  : Decimal
    
    _is_primitive: bool
    _variables   : set[Value]
    _expression  : Expression
    
    @classmethod
    def _init_by_expression(cls, expression:Expression, variables_seen:set[Value]) -> Value:
        variables_seen = variables_seen or set()
        expression = expression.simplify()
        
        # debug #
        if PRINT_MATH_DEBUG:
            print( "Expression:", str(expression) )
            print( "Derivatives:" )
            print( *[f"\t{var.get_id()}:\t{str(expression.differentiate(var))}" for var in variables_seen], sep='\n', end='\n\n' )

        error_sqr = Decimal('0')
        prec = None
        if variables_seen:
            error_sqr = sum( map( lambda var: (expression.differentiate(var).value() * var._error)**2, variables_seen) )
            prec_list = map(lambda var: var.__prec, variables_seen)
            prec      = max( prec_list ) if Value.PREC_MODE == Precision_mode.TOWARDS_MAX else min( prec_list )
        
        new_value               = Value(expression.value(), error_sqr.sqrt())
        new_value.__prec        = prec or new_value.__prec
        new_value._expression   = expression
        new_value._variables    = variables_seen
        new_value._is_primitive = False

        return new_value
    
    @property
    def _value(self) -> Decimal:
        '''
        internally used for calculations and error propagation

        ---
        raw non rounded `value`
        '''
        return Value.shift_separator_left( self.__value, self.__exp )
    
    @property
    def _error(self) -> Decimal:
        '''
        internally used for calculations and error propagation

        ---
        raw non rounded `error`
        '''
        return Value.shift_separator_left( self.__error, self.__exp )
    
    @property
    def _relative_error(self) -> Decimal:
        '''
        internally used for calculations and error propagation

        ---
        raw non rounded `relative_error`
        '''
        return abs( self._error / self._value )
    
    def __round_shift(self, value:Decimal, exp:int=None) -> Decimal:
        '''returns the given value rounded to the set precision'''
        exp = exp if exp is not None else self.__exp
        
        #!!! BE AWARE THAT THIS CAN BREAK/WORK INCORRECTLY ANY TIME FOR VARIOUS REASONS !!!
        # it is necessary to make this transformation to ensure that all values are rounded correctly
        # e.g.
        # let x = Decimal( 20.455 )
        # x than is not "20.455" but "20.454999999999998294697434175759553909301757812..."
        # rounding that to 2 decimals would return 20.45 
        # but it should actually be 20.46
        correctly_rounded_value = Decimal( str(round( float(value), self.__prec )) )
        
        return Value.shift_separator_left( correctly_rounded_value, exp )
    
    @staticmethod
    def shift_separator_left( value:Decimal, places:int ) -> Decimal:
        '''alias for `value.scaleb( places )`'''
        assert isinstance( places, int ), TypeError("places must be of type int")
        return value.scaleb( places )
    
    
    def __str__(self) -> str:
        # TODO: enhance best_exp finding for Zero or numerically zero error values
        # BUG:  trailing zeros are not printed and can cause confusion
        
        # TOLERATE_TILL_EXP = 0 # not used atm.
        
        best_exp = int(math.copysign( 3 * math.floor( self.__exp / 3 ), self.__exp ))
        
        s = f"{Value.shift_separator_left(self.value, -best_exp):f} ± {Value.shift_separator_left(self.error, -best_exp):f}"
        
        if best_exp != 0:
            s = f"({s:s})e{best_exp:+d}"
        
        if Value.PRINT_MODE == Print_mode.WITH_RELATIVE_ERROR:
            s = f"{s:s}    {self.str_relative_error()}"
        
        return s

    def __repr__(self) -> str:
        return f"Value( {self._value}, {self._error}, prec={self.__prec}, id={self.__id} )"
    
    def __hash__(self) -> int:
        return hash( (self.__value, self.__error, self.__exp, self.__prec ) )
    
    def get_id(self) -> str:
        return self.__id if self.__id is not None else repr(self)
    
    
    def _operation_setup(self, _o:Value|Decimal|int|float|str) -> tuple[set[Value], Expression]:
        vars = self._variables
        _o_expression = Singleton(_o)
        
        match _o:
            case Value():
                vars = vars.union( _o._variables )

                if not _o._is_primitive:
                    _o_expression = _o._expression
            
            case Decimal() | int() | float() | str():
                pass
            
            case _:
                raise TypeError( f"the type `{type(_o)}` is not compatible with `Value` types " )
        
        return vars, _o_expression
    
    #----------------------------------------------------------------------------------------------
    # calculation dunder functions
    #----------------------------------------------------------------------------------------------
    def __eq__(self, _o:Value|Any) -> bool:
        return self.__hash__() == hash(_o)
    
    def __pos__(self) -> Value:
        return self
    def __neg__(self) -> Value:
        return Value._init_by_expression(
            Negative( self._expression ),
            self._variables
        )
    
    def __add__(self, _o:Value|Decimal|int|float|str, /) -> Value:
        vars, _o_expression = self._operation_setup( _o )

        return Value._init_by_expression(
            Add( 
                self._expression, 
                _o_expression
                ),
            vars
        )
        
        # _o = Value.to_Value( _o )
        
        # v = self._value + _o._value
        
        # match Value.PROPAGATION_MODE:
        #     case Propagation_mode.GAUSSIAN_ERROR:
        #         e = ( self._error**2 + _o._error**2 ).sqrt()
        #     case Propagation_mode.GREATES_ERROR:
        #         e = self._error + _o._error
        #     case _:
        #         raise ValueError(f"Invalide Value.PROPAGATION_MODE was set, only valid modes are {[m.name for m in Propagation_mode]}")
    __radd__ = __add__
    
    def __sub__(self, _o:Value|Decimal|int|float|str, /) -> Value:
        return self + ( -_o )
    def __rsub__(self, _o:Value|Decimal|int|float|str, /) -> Value:
        return _o + ( -self )

    def __mul__(self, _o:Value|Decimal|int|float|str, /) -> Value:
        vars, _o_expression = self._operation_setup( _o )

        return Value._init_by_expression(
            Multiply( 
                self._expression, 
                _o_expression
                ),
            vars
        )
        
        # _o = Value.to_Value( _o )
        
        # v = self._value * _o._value
        
        # match Value.PROPAGATION_MODE:
        #     case Propagation_mode.GAUSSIAN_ERROR:
        #         e = ( self._relative_error**2 + _o._relative_error**2 ).sqrt()
        #     case Propagation_mode.GREATES_ERROR:
        #         e = self._relative_error + _o._relative_error
        #
        # e = abs(v) * e
    __rmul__ = __mul__
    
    def __truediv__(self, _o:Value|Decimal|int|float|str, /) -> Value:
        vars, _o_expression = self._operation_setup( _o )

        return Value._init_by_expression(
            Fraction( 
                self._expression, 
                _o_expression
                ),
            vars
        )
        
        # _o = Value.to_Value( _o )
        
        # v = self._value / _o._value
        
        # match Value.PROPAGATION_MODE:
        #     case Propagation_mode.GAUSSIAN_ERROR:
        #         e = ( self._relative_error**2 + _o._relative_error**2 ).sqrt()
        #     case Propagation_mode.GREATES_ERROR:
        #         e = self._relative_error + _o._relative_error
        #     case _:
        #         raise ValueError(f"Invalide Value.PROPAGATION_MODE was set, only valid modes are {[m.name for m in Propagation_mode]}")
        
        # e = abs(v) * e
    def __rtruediv__(self, _o:Value|Decimal|int|float|str, /) -> Value:
        vars, _o_expression = self._operation_setup( _o )

        return Value._init_by_expression(
            Fraction( 
                _o_expression, 
                self._expression
                ),
            vars
        )
    
    def __pow__(self, exp:Value|Decimal|int|float, _mod:None=None) -> Value:
        assert _mod is None, NotImplementedError("modulo operations are not supported")
        
        vars, _o_expression = self._operation_setup( exp )

        return Value._init_by_expression(
            Pow( 
                self._expression, 
                _o_expression,
                ),
            vars
        )
        
        # v = self._value ** exp._value
        
        # match Value.PROPAGATION_MODE:
        #     case Propagation_mode.GAUSSIAN_ERROR:
        #         e = ( (exp._value * self._relative_error)**2 + (self._value.ln() * exp._error)**2 ).sqrt()
        #     case Propagation_mode.GREATES_ERROR:
        #         e = abs(exp._value * self._relative_error) + abs(self._value.ln() * exp._error)
        #     case _:
        #         raise ValueError(f"Invalide Value.PROPAGATION_MODE was set, only valid modes are {[m.name for m in Propagation_mode]}")
        
        # return Value( v, v*e, prec=prec )
    def __rpow__(self, base:Value|Decimal|int|float, _mod:None=None) -> Value:
        assert _mod is None, NotImplementedError("modulo operations are not supported")
        
        vars, _o_expression = self._operation_setup( base )

        return Value._init_by_expression(
            Pow( 
                _o_expression,
                self._expression, 
                ),
            vars
        )
        
        # return Value._init_by_expression(
        #     Pow( 
        #         Singleton( base ), 
        #         self._expression 
        #         ),
        #     self.__variables.union( [ base ] ) if isinstance(base, Value) else self.__variables
        # )

class Measurement():
    """creating statistics to a set of measurements"""
    
    __global_measurement_count: int = 0
    __id: Optional[str]
    
    N: int
    spanwidth: float
    
    average: Value
    median : Value
    
    sqr_sigma: Value
    sigma    : Value
    
    sqr_s: Value
    s    : Value
    
    exp: PF
    
    def __init__(self, data:Sequence[Decimal|int|float|str], exp:int|PF=0, force_prec=2, *, id:Optional[str]=None) -> None:
        """
        Args:
            data (`Sequence[Decimal | int | float]`): data to be analyzed
            exp (`int` | `PF`, optional): shift exponent of the given value and error, same as multiplying the given value and error by 10**exp. Defaults to 0.
            force_prec (`int`, optional): amount of significant digits used for representation. Defaults to 2.
            id (`Optional[str]`): same as `Value(..., id)`. If pressend the `id` string is being supplied to all `Value` attributes of this `Measurement` instance in the form of `$id$_$name_of_attribute$`. If `id` is None the `id` is set to `measurement_$some_number$`
        """
        self.__id = id if id else f"measurement_{Measurement.__global_measurement_count:d}"
        Measurement.__global_measurement_count += 1
        
        data = sorted( [Decimal(d) for d in data] )
        
        self.exp       = PF( exp )
        self.N         = len( data )
        self.spanwidth = (data[-1] - data[0]) * self.exp.factor
        avg            = sum( data ) / self.N
        
        temp_variance = sum( map( lambda x: (x-avg)**2, data ) )
        
        sqr_sigma = temp_variance / self.N
        sqr_s     = temp_variance / ( self.N - 1 )
        
        if self.N % 2 == 0:
            self.median = Value( ( data[self.N//2] + data[self.N//2+1] ) / 2, '0.0', exp=self.exp, prec=force_prec, id=f'{self.__id}_median' )
        else:
            self.median = Value( data[self.N//2], '0', exp=self.exp, prec=force_prec, id=f'{self.__id}_median' )
        
        self.average = Value( avg, (sqr_s / self.N)**0.5, exp=self.exp, prec=force_prec, id=f'{self.__id}_avg' )
        
        self.sqr_sigma = Value( sqr_sigma, sqr_sigma / ( 2 * self.N ), exp=self.exp, prec=force_prec, id=f'{self.__id}_sqr_sigma' )
        self.sqr_s     = Value( sqr_s, sqr_s / ( 2 * ( self.N - 1 ) ), exp=self.exp, prec=force_prec, id=f'{self.__id}_sqr_s' )
        
        self.sigma = Value( sqr_sigma**0.5, (sqr_sigma / ( 2 * self.N ))**0.5, exp=self.exp, prec=force_prec, id=f'{self.__id}_sigma' )
        self.s     = Value( sqr_s**0.5, ( sqr_s / ( 2 * ( self.N - 1 ) ) )**0.5, exp=self.exp, prec=force_prec, id=f'{self.__id}_s' )
    
    def __str__(self) -> str:
        width = 50
        
        print_mode = Value.PRINT_MODE
        Value.PRINT_MODE = Print_mode.NO_RELATIVE_ERROR
        
        fmt = "| {:>9s}: {:^%ds} {:<11s} |" % (width - 9 - 2 - 1 - 11)
        fmt_value = lambda name, value: fmt.format( name, str(value), value.str_relative_error() )
        
        bound_top_bottom = "+-" + '-'*width + "-+"
        h_line           = '| ' + '-'*width + ' |'
        
        s = '\n'.join( [
            bound_top_bottom,
            
            fmt.format( 'N', str(self.N), '' ),
            fmt.format( 'Spanwidth', str(round(self.spanwidth / self.exp.factor, 10)) +' * '+ str(self.exp), '' ),
            
            h_line,
            
            fmt_value( 'Average', self.average ),
            fmt_value( 'Median' , self.median  ),
            
            h_line,
            
            fmt_value( 'σ²', self.sqr_sigma ),
            fmt_value( 'σ' , self.sigma     ),
            fmt_value( 's²', self.sqr_s     ),
            fmt_value( 's' , self.s         ),
            
            bound_top_bottom,
        ] )
        
        Value.PRINT_MODE = print_mode
        
        return s

class Regression_linear_simple():
    N: int
    X: list[ Decimal ]
    Y: list[ Decimal ]
    
    
    sum_x: Decimal
    '''∑ ( `xₖ` )'''

    sum_y: Decimal
    '''∑ ( `yₖ` )'''

    sum_x_sqr: Decimal
    '''∑ ( `xₖ`² )'''

    sum_x_y: Decimal
    '''∑ ( `xₖ`*`yₖ` )'''

    determinate_s: Decimal
    '''`∆` = `N` * ∑ ( `xₖ`² ) - (∑ ( `xₖ` ))²'''
    
    coef_A: Value
    '''a = 1/`∆` * ( ∑ ( `xₖ`² ) * ∑ ( `yₖ` ) - ∑ ( `xₖ` ) * ∑ ( `xₖ`*`yₖ` ) )'''
    coef_B: Value
    '''b = 1/`∆` * ( `N` * ∑ ( `xₖ`*`yₖ` ) - ∑ ( `xₖ` ) * ∑ ( `yₖ` ) )'''
    
    s_sqr  : Decimal
    
    def __init__(self, x_data:Sequence[Decimal|int|float|str], y_data:Sequence[Decimal|int|float|str]) -> None:
        x_data = list( x_data )
        y_data = list( y_data )
        
        assert all( map( lambda d: isinstance( d, (Decimal, int, float, str)), x_data ) ), TypeError( "Some entries in x_data are not of type (Decimal | int | float | str)" )
        assert all( map( lambda d: isinstance( d, (Decimal, int, float, str)), y_data ) ), TypeError( "Some entries in y_data are not of type (Decimal | int | float | str)" )
        assert len(x_data) == len(y_data), ValueError( f"Sequences x_data of length {len(x_data):d} and y_data of length {len(y_data):d} must be of equal length" )
        
        self.X = [ Decimal(d) for d in x_data ]
        self.Y = [ Decimal(d) for d in y_data ]
        self.N = len( self.X )
        
        self.sum_x     = sum( self.X )
        self.sum_y     = sum( self.Y )
        self.sum_x_sqr = sum( map( lambda x: x*x, self.X ) )
        self.sum_x_y   = sum( map( lambda x,y: x*y, self.X, self.Y ) )
        
        self.determinate_s = self.N * self.sum_x_sqr - self.sum_x**2
        
        a = ( self.sum_x_sqr*self.sum_y - self.sum_x*self.sum_x_y ) / self.determinate_s
        b = ( self.N*self.sum_x_y - self.sum_x*self.sum_y ) / self.determinate_s
        
        self.s_sqr = sum( map( lambda x, y: (y-a-b*x)**2, self.X, self.Y) ) / (self.N-2)
        
        sigma_a = math.sqrt(          self.s_sqr/self.determinate_s * self.sum_x_sqr )
        sigma_b = math.sqrt( self.N * self.s_sqr/self.determinate_s )

        self.coef_A = Value( a, sigma_a )
        self.coef_B = Value( b, sigma_b )
    
    def __str__(self) -> str:
        width = 50

        max_name_width = 9
        
        print_mode = Value.PRINT_MODE
        Value.PRINT_MODE = Print_mode.NO_RELATIVE_ERROR
        
        fmt = lambda format_type: "| {:>%ds}: {:^%d%s} |" % (max_name_width, width - max_name_width - 2, format_type)
        fmt_center  = "| {:^%ds} |" % width
        fmt_decimal = lambda name, value: fmt('f').format( name, float(value) )
        
        bound_top_bottom = "+-" + '-'*width + "-+"
        h_line           = '| ' + '-'*width + ' |'
        
        s = '\n'.join( [
            bound_top_bottom,
            
            fmt_center.format( f"Y = ({self.coef_A}) + ({self.coef_B})*X" ),
            
            h_line,
            
            fmt('d').format( 'N', self.N ),
            
            fmt_decimal( "∑ Xₖ"    , self.sum_x    ),
            fmt_decimal( "∑ Yₖ"    , self.sum_y    ),
            fmt_decimal( "∑ Xₖ²"   , self.sum_x_sqr),
            fmt_decimal( "∑ Xₖ*Yₖ", self.sum_x_y  ),
            
            h_line,
            
            fmt_decimal( "∆" , self.determinate_s),
            fmt_decimal( "s²", self.s_sqr        ),
            fmt_decimal( "s" , self.s_sqr.sqrt() ),
            
            bound_top_bottom,
        ] )
        
        s += '\n\n'
        
        
        y_regression = lambda x: self.coef_A.value + self.coef_B.value*x
        
        
        s += f"{'xₖ':^4s} | {'yₖ':^5s} | {'y':^5s} | {'(y - yₖ)²':s}\n"
        s += '-'*(4+3+5+3+5+3+12) + '\n'
        
        sum_delta = Decimal('0')
        for x, y in zip(self.X, self.Y):
            y_of_x = y_regression(x)
            sum_delta += (y_of_x - y)**2
            
            s += f"{x:^.2f} | {y:^.2f} | {y_of_x:^.2f} | {(y_of_x - y)**2:^.2f}\n"
        
        s += '-'*(4+3+5+3+5+3+12) + '\n'
        s += f"∑ (y-yₖ)²: {sum_delta:.2f}"
        
        
        Value.PRINT_MODE = print_mode
        
        return s

class Regression_linear():
    N: int
    X: list[ Decimal ]
    Y: list[ Value ]
    
    sum_1: Decimal
    '''∑ ( 1/`σₖ`²)'''

    sum_x: Decimal
    '''∑ ( `xₖ`/`σₖ`²)'''
    
    sum_x_sqr: Decimal
    '''∑ ( `xₖ`²/`σₖ`²)'''
    
    sum_y: Decimal
    '''∑ ( `yₖ`/`σₖ`²)'''
    
    sum_x_y  : Decimal
    '''∑ ( `xₖ`*`yₖ`/`σₖ`²)'''
    
    determinate: Decimal
    '''`∆` = ∑ ( 1/`σₖ`²) * ∑ ( `xₖ`²/`σₖ`²) - (∑ ( `xₖ`/`σₖ`²))²'''
    
    coef_A: Value
    '''a = 1/`∆` * ( ∑ ( `yₖ`/`σₖ`²) * ∑ ( `xₖ`²/`σₖ`²) - ∑ ( `xₖ`/`σₖ`²) * ∑ ( `xₖ`*`yₖ`/`σₖ`²) )'''
    coef_B: Value
    '''b = 1/`∆` * ( ∑ ( 1/`σₖ`²) * ∑ ( `xₖ`*`yₖ`/`σₖ`²) - ∑ ( `xₖ`/`σₖ`²) * ∑ ( `yₖ`/`σₖ`²) )'''
    
    def __init__(self, x_data:Sequence[Decimal|int|float|str], y_data:Sequence[Value]) -> None:
        x_data = list( x_data )
        y_data = list( y_data )
        
        assert all( map( lambda d: isinstance( d, (Decimal, int, float, str)), x_data ) ), TypeError( "Some entries in x_data are not of type (Decimal | int | float | str)" )
        assert all( map( lambda d: isinstance( d, Value), y_data ) ), TypeError( "Some entries in y_data are not of type Value" )
        assert len(x_data) == len(y_data), ValueError( f"Sequences x_data of length {len(x_data):d} and y_data of length {len(y_data):d} must be of equal length" )
        
        self.X = [ Decimal(d) for d in x_data ]
        self.Y = [ d for d in y_data ]
        
        self.N = len( self.X )
        
        self.sum_1     = Decimal('0')
        self.sum_x     = Decimal('0')
        self.sum_x_sqr = Decimal('0')
        self.sum_y     = Decimal('0')
        self.sum_x_y   = Decimal('0')
        for x, y in zip(self.X, self.Y):
            assert y.error != Decimal('0'), f"Data point (x= {x}, y= {y}) is invalid because the error of y is zero"
            _1_err_sqr = y.error**-2
            
            self.sum_1     += _1_err_sqr
            self.sum_x     += _1_err_sqr * x
            self.sum_x_sqr += _1_err_sqr * x*x
            self.sum_y     += _1_err_sqr * y.value
            self.sum_x_y   += _1_err_sqr * x * y.value
        
        self.determinate = self.sum_1 * self.sum_x_sqr - self.sum_x * self.sum_x
        
        a = ( self.sum_y * self.sum_x_sqr - self.sum_x * self.sum_x_y ) / self.determinate
        b = ( self.sum_1 * self.sum_x_y   - self.sum_x * self.sum_y   ) / self.determinate
        
        sigma_a = math.sqrt( self.sum_x_sqr / self.determinate )
        sigma_b = math.sqrt( self.sum_1     / self.determinate )

        self.coef_A = Value( a, sigma_a )
        self.coef_B = Value( b, sigma_b )
    
    def __str__(self) -> str:
        width = 50

        max_name_width = 12
        
        print_mode = Value.PRINT_MODE
        Value.PRINT_MODE = Print_mode.NO_RELATIVE_ERROR
        
        fmt = lambda format_type: "| {:>%ds}: {:^%d%s} |" % (max_name_width, width - max_name_width - 2, format_type)
        fmt_center  = "| {:^%ds} |" % width
        fmt_decimal = lambda name, value: fmt('f').format( name, float(value) )
        
        bound_top_bottom = "+-" + '-'*width + "-+"
        h_line           = '| ' + '-'*width + ' |'
        
        s = '\n'.join( [
            bound_top_bottom,
            
            fmt_center.format( f"Y = {self.coef_A} + ({self.coef_B})*X" ),
            
            h_line,
            
            fmt('d').format( 'N', self.N ),
            
            fmt_decimal( "∑    Xₖ/σ²", self.sum_x    ),
            fmt_decimal( "∑    Yₖ/σ²", self.sum_y    ),
            fmt_decimal( "∑   Xₖ²/σ²", self.sum_x_sqr),
            fmt_decimal( "∑ Xₖ*Yₖ/σ²", self.sum_x_y  ),
            
            h_line,
            
            fmt_decimal( "∆", self.determinate  ),
            
            bound_top_bottom,
        ] )
        
        s += '\n\n'
        
        Value.PRINT_MODE = print_mode
        
        return s


def transform_to(
    x_data              : Sequence[Value|Decimal|int|float|str],
    y_data              : Sequence[Value|Decimal|int|float|str],
    transform_function_x: Callable[[Value|Decimal|int|float|str], Value|Decimal] = Decimal,
    transform_function_y: Callable[[Value|Decimal|int|float|str], Value|Decimal] = Decimal,
    round_to            : int = 3
    ) -> tuple[list[Value|Decimal|int|float|str], list[Value|Decimal|int|float|str]]:
    """
    Transform a given point-cloud via the supplied transform_function's

    additionally prints a table with the transformed data

    Args:
        x_data (`Sequence[Value | Decimal | int | float | str]`): x coordinates of point cloud
        y_data (`Sequence[Value | Decimal | int | float | str]`): y coordinates of point cloud
        transform_function_x (`Callable[[Value | Decimal | int | float | str], Value | Decimal]`): function to be applied to every `x_data` entry. Defaults to `lambda x: Decimal(x)`.
        transform_function_y (`Callable[[Value | Decimal | int | float | str], Value | Decimal]`): function to be applied to every `y_data` entry. Defaults to `lambda x: Decimal(x)`.
        round_to (`int`, optional): [!ONLY FOR PRINTING!] post-decimal places all values are rounded to. Defaults to 3.

    Returns:
        `tuple[list[Decimal], list[Value]|list[Decimal|int|float|str]]`: transformed point-cloud as tuple[ list_of_transformed_x_data, list_of_transformed_y_data ]
    """
    
    x_data = list( x_data )
    y_data = list( y_data )
    
    assert all( map( lambda d: isinstance( d, (Value, Decimal, int, float, str)), x_data ) ), TypeError( "Some entries in x_data are not of type (Value | Decimal | int | float | str)" )
    assert all( map( lambda d: isinstance( d, (Value, Decimal, int, float, str)), y_data ) ), TypeError( "Some entries in y_data are not of type (Value | Decimal | int | float | str)" )
    assert len(x_data) == len(y_data), ValueError( f"Sequences x_data of length {len(x_data):d} and y_data of length {len(y_data):d} must be of equal length" )
    
    transform_function_x = transform_function_x if isinstance(transform_function_x, Callable) else Decimal
    transform_function_y = transform_function_y if isinstance(transform_function_y, Callable) else Decimal
    
    transformed_x = [ transform_function_x(x) for x in x_data ]
    transformed_y = [ transform_function_y(y) for y in y_data ]
    
    print( "transformed x_data | transformed y_data" )
    
    for x, y in zip( transformed_x, transformed_y ):
        print(
            ("{:>18.%d}" % round_to).format( x ) if isinstance(x, Decimal) else f"{str(x):18}",
            ' | ',
            ("{:>18.%d}" % round_to).format( y ) if isinstance(y, Decimal) else f"{str(y):18}",
            sep=''
            )
    
    return transformed_x, transformed_y


if __name__ == '__main__':
    # Übung 6 
    # PREC = 2
    # a = Value( '1.630' , '0.021' , prec=PREC, id='a' )
    # b = Value( '0.6649', '0.0040', prec=PREC, id='b' )
    # x = Value( '1.376' , '0.037' , prec=PREC, id='x' )
    
    # print(a.e, a.re)
    
    # print( f"a = {a}" )
    # print( f"b = {b}" )
    # print( f"x = {x}" )
    
    # y1 = 13*a*x + 14*a*b*x**2 + 21*a*b**3
    # y2 = exp( ( a - x ) / x )
    # y3 = b * sin( a * x )
    # y4 = ( x - a ) / ( x + b )

    # print( y1 )
    # print( y2 )
    # print( y3 )
    # print( y4 )
    
    # y1.print_info_equation()
    # y2.print_info_equation()
    # y3.print_info_equation()
    # y4.print_info_equation()
    
    
    
    # # Übung 6
    # P = 2
    
    # m_sm   = Value( 30.9829, 0.0092, id='m_sm  ', exp=PF.m.to(PF.k), prec=P )                      #       mg -> kg
    # m_so   = Value( 6.8245 , 0.0035, id='m_so  ', exp=PF.m.to(PF.k), prec=P )                      #       mg -> kg
    # H      = Value( 33.75  , 0.37  , id='H     ', exp=PF.c, prec=P )                               #       cm -> m
    # t      = Value( 76.34  , 0.74  , id='t     ', exp=PF.NONE, prec=P )                            #      sec -> sec
    # rho_fl = Value( 0.962  , 0.000 , id='rho_fl', exp=(PF._/PF.c**3).to(PF.k/PF._**3), prec=P )    # g / cm^3 -> kg / m^3
    
    # d = Measurement( [1.249, 1.234, 1.252, 1.238, 1.235, 1.246, 1.262, 1.255, 1.243], PF.m, P, id='d' )
    
    
    # m_7k = m_sm - m_so
    # m_k  = m_7k / 7
    
    # r_k = d.average / 2
    
    # V_k = 4/3 * math.pi * r_k**3
    
    # rho_k = m_k / V_k
    
    # rho_d = rho_k-rho_fl
    
    # nu = 2/9 * 9.810 * rho_d * r_k**2 * t / H
    
    # print( f"m_sm   = {m_sm  }" )
    # print( f"m_so   = {m_so  }" )
    # print( f"H      = {H     }" )
    # print( f"t      = {t     }" )
    # print( f"rho_fl = {rho_fl}" )
    # print( f"d      = {d.average}" )
    # print()
    # print( f"Measurement:\n{d}" )
    # print()
    # print( f"m_7k  = {m_7k }" )
    # print( f"m_k   = {m_k  }" )
    # print( f"r_k   = {r_k  }" )
    # print( f"V_k   = {V_k  }" )
    # print( f"rho_k = {rho_k}" )
    # print( f"rho_d = {rho_d}" )
    # print( f"nu    = {nu   }" )
    # print()
    
    # nu.print_info_equation()
    
    
    # Übung 8
    # test data
    # X = [0.20, 1.17, 1.96, 2.98, 4.05, 5.12, 5.93, 7.01, 8.24]
    # Y = [52.4, 54.6, 57.7, 59.5, 62.7, 65.6, 67.0, 69.0, 72.8]
    
    # X = [0.16, 1.19, 2.12, 3.11, 4.10, 4.83, 6.12, 7.01, 8.19]
    # Y = [54.7, 54.8, 57.8, 60.3, 63.1, 63.7, 68.3, 71.0, 72.2]
    
    # print(Regression_linear_simple(X, Y))
    
    
    # Übung 9
    T = [0.56, 1.23, 1.48, 2.07, 2.91, 3.43, 4.23, 4.87, 5.73, 6.36]
    U = [53.05, 38.64, 35.16, 26.57, 17.72, 14.25, 10.27, 7.37, 4.67, 4.28]
    R = 3.40 * PF.M
    
    
    U = map( lambda u: Value( u, 0.5 ), U )
    
    T, U = transform_to( T, U, transform_function_y=lambda y: log(y, '10'), round_to=3 )
    
    regression = Regression_linear(T, U)
    
    U_0 = Decimal('10') ** regression.coef_A
    tau = - log( exp('1'), '10' ) / regression.coef_B
    C = tau * 60 / R
    
    print(regression)
    print( f"U_0 = {U_0}" )
    print( f"tau = {tau}" )
    print( f"C   = {C}" )
    print()
    print( f"tau2 =", tau2:=log( exp('1'), '10' ) * -2.7 / log(Value(20.0, 0.2)/Value(68.5, 0.5), '10') )
    print( f"C2   = { tau2 * 60 / R }" )
    print()
    
    print( f"U0/e       = {U_0/exp('1')}" )
    print( f"tau(U0/e)  = { (log(U_0/exp('1'), '10')-regression.coef_A)/(regression.coef_B * 1)}" )
    print( f"U0/e²      = {U_0/exp('2')}" )
    print( f"tau(U0/e)  = { (log(U_0/exp('2'), '10')-regression.coef_A)/(regression.coef_B)}" )
    
    pass