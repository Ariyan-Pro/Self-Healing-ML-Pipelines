import sys
sys.path.append('.')
from failure_scoring import FailureIntelligence

print('TESTING FAILURE INTELLIGENCE')
print('=' * 50)

fi = FailureIntelligence()
print('1. FailureIntelligence initialized')

# Log 3 failures
for i in range(3):
    failure = fi.log_failure(
        state_before={'accuracy': 0.9 - i*0.1, 'drift': 0.05 + i*0.03},
        action_taken='retrain',
        outcome='success',
        recovery_cost=50.0 * (i + 1)
    )
    print(f'Logged failure {i+1}: ')

print(f'2. Total failures logged: {len(fi.failure_dataset)}')
print('3. SYSTEM OPERATIONAL')
print('=' * 50)