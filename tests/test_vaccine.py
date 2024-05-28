"""
Run tests of vaccines
"""

# %% Imports and settings
import starsim as ss
import numpy as np

def run_test_sir_vaccine( efficacy, all_or_nothing ):
  
  # parameters
  v_frac      = 0.5    # fraction of population vaccinated
  total_cases = 500    # total cases at which point we check results
  tol         = 3      # tolerance in standard deviations for simulated checks
  
  # create a basic SIR sim
  sim = ss.Sim(
    n_agents = 10000,     
    pars    = dict(
      networks = dict(     
          type = 'random', 
          n_contacts = 4    
      ),
      diseases = dict(      
          type      = 'sir',     
          init_prev = 0.01,  
          dur_inf   = 0.1,
          p_death   = 0,
          beta      = 6,       
      )
    ),
    n_years = 10,
    dt      = 0.01
  )
  sim.initialize( verbose = False )
  
  # work out who to vaccinate
  in_trial = sim.people.sir.susceptible.uids
  n_vac    = round( len( in_trial ) * v_frac  )
  in_vac   = np.random.choice( in_trial, n_vac, replace = False )
  in_pla   = np.setdiff1d( in_trial, in_vac )
  uids     = ss.uids( in_vac )

  # create and apply the vaccination
  vac = ss.sir_vaccine(  efficacy = efficacy, all_or_nothing = all_or_nothing )
  vac.initialize( sim )
  vac.administer( sim.people, uids )
  
  # check the relative susceptibility at the start of the simulation
  rel_susc = sim.people.sir.rel_sus.values
  assert min( rel_susc[ in_pla ] ) == 1.0, f'Placebo arm is not fully susceptible'
  if all_or_nothing :
    assert min( rel_susc[ in_vac ] ) == 0.0, f'Nobody fully vaccinated (all_or_nothing)'
    assert max( rel_susc[ in_vac ] ) == 1.0, f'Vaccine effective in everyone (all_or_nothing)'
    mean = n_vac * ( 1 - efficacy )
    sd   = np.sqrt( n_vac * efficacy * ( 1 - efficacy ) )
    assert ( np.mean( rel_susc[ in_vac] ) - mean ) / sd < tol, f'Incorrect mean susceptibility in vaccinated (all_or_nothing)'
  else : 
    assert max( abs( rel_susc[ in_vac ] - ( 1 - efficacy ) ) ) < 0.0001 , f'Relative susceptibility not 1-efficacy (leaky)'
  
  # run the simulation until sufficient cases
  old_cases = []
  for idx in range( 1000 ) :
    sim.step()
    susc  = sim.people.sir.susceptible.uids
    cases = np.setdiff1d( in_trial, susc )
    if  len( cases ) > total_cases :
      break
    old_cases = cases
 
  if len( cases ) > total_cases :
    cases = np.concatenate( [ old_cases, np.random.choice( np.setdiff1d( cases, old_cases ), total_cases - len( old_cases ), replace = False ) ] )
  vac_cases = np.intersect1d( cases, in_vac ) 

  # check to see whether the number of cases are as expected  
  p    = v_frac * ( 1 - efficacy ) / ( 1 - efficacy * v_frac )
  mean = total_cases * p 
  sd   = np.sqrt( total_cases * p * ( 1 - p ) ) 
  assert ( len( vac_cases ) - mean ) / sd < tol, f'Incorrect proportion of vaccincated infected'
  
  # for all or nothing check that fully vaccinated did not get infectedpyt
  if all_or_nothing :
    assert len( np.intersect1d( vac_cases, in_vac[ rel_susc[ in_vac ] == 1.0 ] ) ) == len( vac_cases ), f'Not all vaccine cases amongst vaccine failures (all or nothing)'
    assert len( np.intersect1d( vac_cases, in_vac[ rel_susc[ in_vac ] == 0.0 ] ) ) == 0, f'Vaccine cases amongst fully vaccincated (all or nothing)'
   
  return sim

def test_sir_vaccine_leaky():
  run_test_sir_vaccine( 0.3, False )

def test_sir_vaccine_all_or_nothing():
  run_test_sir_vaccine( 0.3, True )


if __name__ == '__main__':
    sir_vaccine_leaky   = test_sir_vaccine_leaky()
    sir_vaccine_a_or_n  = test_sir_vaccine_all_or_nothing()
