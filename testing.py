from Fehlerrechnung import *

from typing import Callable
from random import random, randint, choice
from math import ceil, log
from collections import deque

def get_id_set( amount:int ) -> list[str]:
    # A-Z: [65, 90], a-z: [97, 122]
    alphabet = [ chr(i) for i in range(97, 123) ] #+ [ chr(i) for i in range(65, 91) ]

    chars = ceil( log(amount, len(alphabet)) + 1E-7 )
    modulo_at_level = [ len(alphabet)**i for i in range(chars) ]
    
    return [
        ''.join([ 
                 alphabet[ i//modulo_at_level[ chars-1 - ch ] % len(alphabet) ] 
                 for ch in range(chars) 
                 ])
        for i in range( len(alphabet)**chars )
    ]

def get_random_values( amount:int, value_bounds_lower_upper:tuple[float, float]=(0, 1000), relative_error_max:float=0.3, force_precision:int=2, values_with_random_id_percentile:float=0.5 ) -> list[Value]:
    ids = deque( get_id_set( amount ) )
    
    span = value_bounds_lower_upper[1] - value_bounds_lower_upper[0]
    
    out = []
    
    for _ in range(amount):
        val = random()*span + value_bounds_lower_upper[0]
        err = abs(val)*(random()*relative_error_max)

        val_id = None
        if random() <= values_with_random_id_percentile:
            val_id = choice( ids )
            ids.remove( val_id )
        
        out.append( Value( val, err, prec=force_precision, id=val_id ) )
    
    return out

def get_random_decimals( amount:int, bounds_lower_upper:tuple[float, float]=(-100, 100), round_to:int=2 ) -> list[Decimal]:
    span = bounds_lower_upper[1] - bounds_lower_upper[0]
    
    return [
        Decimal( str(round(random()*span + bounds_lower_upper[0], round_to)) )
        for _ in range(amount)
    ]

class op_probabilities():
    neg: float

    add: float
    mul: float
    div: float
    pow: float
    
    exp: float
    log: float
    
    sin   : float
    cos   : float
    tan   : float
    arcsin: float
    arccos: float
    arctan: float
    
    sinh  : float
    cosh  : float
    tanh  : float
    arsinh: float
    arcosh: float
    artanh: float
    
    __ITEMS: dict[str, Callable[[tuple[Value|Decimal, Value|Decimal]| Value | Decimal], Value | Decimal]] = {
        # args = 1
        "neg": lambda x: -x, 
        
        "exp": exp,
        
        "sin"   : sin,
        "cos"   : cos,
        "tan"   : tan,
        "arcsin": arcsin,
        "arccos": arccos,
        "arctan": arctan,
        
        "sinh"  : sinh,
        "cosh"  : cosh,
        "tanh"  : tanh,
        "arsinh": arsinh,
        "arcosh": arcosh,
        "artanh": artanh,
        
        # args = 2
        "add": lambda x, y: x + y,
        "mul": lambda x, y: x * y,
        "div": lambda x, y: x / y,
        "pow": lambda x, y: x ** y,
        
        "log": log,
    }
    __ITEMS_REDUCTION: dict[str, Expression] = {
        "add": lambda x, y: x + y,
        "mul": lambda x, y: x * y,
        "div": lambda x, y: x / y,
        "pow": lambda x, y: x ** y,
    }
    __items_reduction_rescale: float
    
    def __init__(self,
        neg: float=15,
        add: float=30,
        mul: float=10,
        div: float=10,
        pow: float=5,
        
        exp: float=7,
        log: float=3,
        
        sin   : float=5,
        cos   : float=5,
        tan   : float=5,
        arcsin: float=1,
        arccos: float=1,
        arctan: float=1,
        
        sinh  : float=2,
        cosh  : float=2,
        tanh  : float=2,
        arsinh: float=1,
        arcosh: float=1,
        artanh: float=1
                 ):
        
        self.neg = neg
        
        self.add = add
        self.mul = mul
        self.div = div
        self.pow = pow
        
        self.exp = exp
        self.log = log
        
        self.sin    = sin   
        self.cos    = cos   
        self.tan    = tan   
        self.arcsin = arcsin
        self.arccos = arccos
        self.arctan = arctan
        
        self.sinh   = sinh  
        self.cosh   = cosh  
        self.tanh   = tanh  
        self.arsinh = arsinh
        self.arcosh = arcosh
        self.artanh = artanh
    
        sum_points = sum( self )
        
        self[0] = self[0] / sum_points
        
        for i in range(1, len(self.__ITEMS)):
            self[i] = self[i] / sum_points + self[i-1]
        
        self.__items_reduction_rescale = sum( map(lambda n: self[n], self.__ITEMS_REDUCTION) ) / sum_points
    
    def get_random_operation(self, *, force_reduction_operators:bool=False) -> str:
        rand = random()
        
        items = self.__ITEMS
        
        if force_reduction_operators:
            items = self.__ITEMS_REDUCTION 
            rand *= self.__items_reduction_rescale
        
        best_op = None
        for op in items.keys():
            err = self[op] - rand
            
            if err < 0:
                continue
            
            if best_op is None or self[op] < self[best_op]:
                best_op = op
        
        return best_op
    
    def operate(self, args:list[Expression], force_reduction_operators:bool=False) -> list[Expression]:
        exception_counter = 100
        while exception_counter > 0 :
            out_args    : list[Expression] = [ ex for ex in args ]
            refined_args: list[Expression] = []
            
            try:
                op = self.get_random_operation(force_reduction_operators=force_reduction_operators)
                match op:
                    case "add" | "mul" | "div" | "pow" | "log":
                            refined_args.append( out_args.pop( randint( 0, len(out_args)-1 ) ) )
                            refined_args.append( out_args.pop( randint( 0, len(out_args)-1 ) ) )
                    case  "neg" | "exp"\
                        | "sin" | "cos"  | "tan"  | "arcsin" | "arccos" | "arctan"\
                        | "sinh"| "cosh" | "tanh" | "arsinh" | "arcosh" | "artanh":
                            refined_args.append( out_args.pop( randint( 0, len(out_args)-1 ) ) )
                    case _:
                        return out_args
                # print(op, refined_args)
                
                out_args.insert( 0, self.__ITEMS[op]( *refined_args ) )
                return out_args
            
            except (TypeError, ValueError, AssertionError) as e:
                # print(e)
                exception_counter -= 1
        
        raise ValueError()
    
    
    def __setitem__(self, key:int|str, value:float ) -> None:
        setattr( self, key if isinstance(key, str) else list(self.__ITEMS.keys())[key], value )
    
    def __getitem__(self, key:int|str) -> float:
        if key is None:
            return 0.0
        return getattr( self, key if isinstance(key, str) else list(self.__ITEMS.keys())[key], 0.0 )
    
    def __iter__(self):
        for n in self.__ITEMS.keys():
            yield self[n]

def get_random_equation(
    variable_amount:int,
    constant_amount:int,
    variable_multiples: int = 1, 
    operator_probability_lookup:op_probabilities=op_probabilities(),
    variable_bounds_lower_upper :tuple[float, float]=(0, 1000), variable_relative_error_max:float=0.3, variable_force_precision:int=2,
    constants_bounds_lower_upper:tuple[float, float]=(-100, 100), constants_round_to:int=2
    ) -> Expression:
    assert isinstance(variable_amount, int)                  , "variable_amount must be an integer"
    assert isinstance(constant_amount, int)                  , "constant_amount must be an integer"
    assert isinstance(variable_multiples, int)               , "variable_multiples must be an integer"
    assert isinstance(variable_bounds_lower_upper, Sequence) , "variable_bounds_lower_upper must be a tuple of [float, float]"
    assert isinstance(constants_bounds_lower_upper, Sequence), "constants_bounds_lower_upper must be a tuple of [float, float]"
    assert isinstance(variable_force_precision, int)         , "variable_force_precision must be an integer"
    assert isinstance(constants_round_to, int)               , "constants_round_to must be an integer"
    
    assert variable_amount >= 0, "variable_amount must be non negative"
    assert constant_amount >= 0, "constant_amount must be non negative"
    assert variable_amount > 0 or constant_amount > 0, "variable_amount and constant_amount can not be both 0"    
    
    
    values = get_random_values(
        variable_amount,
        variable_bounds_lower_upper,
        variable_relative_error_max,
        variable_force_precision,
        1.0
        )
    
    constants = get_random_decimals(
        constant_amount,
        constants_bounds_lower_upper,
        constants_round_to
        )
    
    vars = values*variable_multiples + constants
    
    while len(vars) > 1:
        vars = operator_probability_lookup.operate( vars, force_reduction_operators= (len(vars) % 5 == 0) )
    
    # print( '='*100 )
    # print( *[ f"{v.get_id()} = {repr(v)}" for v in values ], sep='\n' )
    # print( *constants, sep='\n', end='\n' )
    # print( '-'*100 )
    vars[0].print_info_equation()
    
    return vars[0]._expression.simplify()
    

if __name__ == '__main__':
    # chunk = 26
    # ids = get_id_set(1000)
    # print( *[' '.join(ids[i:i+chunk]) for i in range(0, len(ids), chunk)], sep='\n' )
    # print( len(ids) )
    
    b = Decimal('10')
    x = Value( 1834.8, 4.5, -3, id='x' )
    v = b ** x
    
    print( v )
    print( v._expression )
    print( v._expression.differentiate(x) )
    
    
    # while True:
    #     get_random_equation(
    #         3, 0, 2,
    #         op_probabilities(),
    #         (-10, 10), 0.5, 2,
    #         (-10, 10), 2
    #     )
    #     input()