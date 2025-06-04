import pandas as pd
import numpy as np

np.random.seed(42)

years_of_experience = np.random.randint(1, 11, size=40)

base_salary = 2000
salary = base_salary + (200 * years_of_experience) + np.random.randint(1, 401, size=40)

data = {'Years_of_Experience': years_of_experience, 'Salary': salary}
df = pd.DataFrame(data)

print(df)
df.to_csv('employee_salaries.csv', index=False)