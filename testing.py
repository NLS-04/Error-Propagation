from Fehlerrechnung import *
from random import random, choice
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
        err = val*(random()*relative_error_max)

        val_id = None
        if random() <= values_with_random_id_percentile:
            val_id = choice( ids )
            ids.remove( val_id )
        
        out.append( Value( val, err, prec=force_precision, id=val_id ) )
    
    return out

if __name__ == '__main__':
    # chunk = 26
    # ids = get_id_set(1000)
    # print( *[' '.join(ids[i:i+chunk]) for i in range(0, len(ids), chunk)], sep='\n' )
    # print( len(ids) )
    
    values = [
        Value(11, 1, id='R1'),
        Value(12, 1, id='R2'),
        Value(10, 3, id='R3')
    ]
    print( *values, '', sep='\n' )
    
    print( Value.weighted_average( *values ) )