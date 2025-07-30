# Import PuLP
from pulp import LpMaximize, LpProblem, LpVariable, value

# Create the LP maximization problem
model = LpProblem("Maximize_Profit", LpMaximize)

# Define the decision variables: x = units of A, y = units of B
x = LpVariable("Product_A", lowBound=0, cat='Continuous')
y = LpVariable("Product_B", lowBound=0, cat='Continuous')

# Objective Function: Maximize 30x + 50y
model += 30 * x + 50 * y, "Total_Profit"

# Constraints
model += 1 * x + 2 * y <= 40, "Machine_Hours"
model += 2 * x + 1 * y <= 60, "Labor_Hours"

# Solve the problem
model.solve()

# Output the results
print(f"Status: {model.status}, {LpProblem.resolve(model)}")
print(f"Produce {x.varValue:.2f} units of Product A")
print(f"Produce {y.varValue:.2f} units of Product B")
print(f"Maximum Profit = ₹{value(model.objective):.2f}")
