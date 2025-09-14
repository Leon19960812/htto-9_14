import sys

def main():
    try:
        import cvxpy as cp
        print("cvxpy version:", cp.__version__)
        solvers = set(cp.installed_solvers())
        print("installed solvers:", sorted(solvers))
        if "MOSEK" not in solvers:
            print("ERROR: MOSEK not found in installed solvers.")
            sys.exit(2)

        x = cp.Variable(3)
        obj = cp.Minimize(cp.sum_squares(x - 1))
        cons = [x >= 0, cp.norm2(x) <= 5]
        prob = cp.Problem(obj, cons)
        prob.solve(solver=cp.MOSEK, verbose=True)
        print("status:", prob.status)
        print("optval:", prob.value)
        print("x*:", x.value)
        if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            sys.exit(3)
        return 0
    except Exception as e:
        print("EXCEPTION:", e)
        return 1

if __name__ == "__main__":
    sys.exit(main())

