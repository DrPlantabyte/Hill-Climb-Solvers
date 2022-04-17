package net.plantabyte.hillclimb;

import java.util.Arrays;
import java.util.function.Function;

/**
 * The Hill-Climb parameter optimization algorithm is the classic brute-force
 * solver. It is extremely robust, but is usually slower than other methods.
 */
public class HillClimbSolver {
	private final double precision;
	private final int iterationLimit;
	
	/**
	 * Standard constructor
	 * @param precision The desired precision (epsilon) for parameter optimization
	 * @param iterationLimit The maximum number of iterations to use when optimizing parameters
	 */
	public HillClimbSolver(double precision, int iterationLimit){
		if(precision <= 0.0) throw new IllegalArgumentException("Precision must be greator than zero");
		this.iterationLimit = iterationLimit;
		this.precision = precision;
	}
	
	/**
	 * Constructor with default iteration limit (1 million iterations)
	 * @param precision The desired precision (epsilon) for parameter optimization
	 */
	public HillClimbSolver(double precision){
		this(precision, 1000000);
	}
	
	/**
	 * Optimizes the provided parameter array to maximize the output of the provided function
	 * @param func The scoring function to maximize, which must be able to take
	 *             <code>initialParams</code> as it's input argument
	 * @param initialParams Initial parameter values
	 * @return Optimized parameter values
	 */
	public double[] maximize(
			final Function<double[], Double> func, final double[] initialParams
	) {
		final int numParams = initialParams.length;
		double[] params = Arrays.copyOf(initialParams, numParams);
		double[] jumpSizes = new double[numParams];
		Arrays.fill(jumpSizes, 16*precision);
		int iters = 0;
		double baseVal = func.apply(params);
		do {
			for(int i = 0; i < numParams; i++){
				double[] leftJump = Arrays.copyOf(params, numParams);
				leftJump[i] = leftJump[i] - jumpSizes[i];
				double[] leftLongJump = Arrays.copyOf(params, numParams);
				leftLongJump[i] = leftLongJump[i] - 2*jumpSizes[i];
				double[] rightJump = Arrays.copyOf(params, numParams);
				rightJump[i] = rightJump[i] + jumpSizes[i];
				double[] rightLongJump = Arrays.copyOf(params, numParams);
				rightLongJump[i] = rightLongJump[i] + 2*jumpSizes[i];
				double[][] pArray = {params, leftJump, rightJump, leftLongJump, rightLongJump};
				double[] valArray = {
						baseVal, func.apply(leftJump), func.apply(rightJump),
						func.apply(leftLongJump), func.apply(rightLongJump)
				};
				int bestIndex = indexOfMax(valArray);
				baseVal = valArray[bestIndex];
				params = pArray[bestIndex];
				if(bestIndex == 0){
					// existing param already best, shrink step size
					jumpSizes[i] = jumpSizes[i] * 0.25;
				} else if(bestIndex > 2){
					// long jump gave best result, expand step size
					jumpSizes[i] = jumpSizes[i] * 4;
				}
			}
		} while(iters++ < iterationLimit && max(jumpSizes) > precision);
		return params;
	}
	/**
	 * Optimizes the provided parameter array to minimize the output of the provided function
	 * @param func The scoring function to minimize, which must be able to take
	 *             <code>initialParams</code> as it's input argument
	 * @param initialParams Initial parameter values
	 * @return Optimized parameter values
	 */
	public double[] minimize(Function<double[], Double> func, double[] initialParams){
		Function<double[], Double> minus = (double[] params) -> -1 * func.apply(params);
		return maximize(minus, initialParams);
	}
	
	/**
	 * Like Math.max(a, b), but for arrays
	 * @param darr array of values
	 * @return highest value in the array
	 */
	protected static double max(double[] darr){
		double m = darr[0];
		for(int i = 1; i < darr.length; i++){
			double n = darr[i];
			if(n > m){
				m = n;
			}
		}
		return m;
	}
	
	/**
	 * Returns the index of the highest value in the array
	 * @param darr array of values
	 * @return index in the array
	 */
	protected static int indexOfMax(double[] darr){
		double m = darr[0];
		int index = 0;
		for(int i = 1; i < darr.length; i++){
			double n = darr[i];
			if(n > m){
				m = n;
				index = i;
			}
		}
		return index;
	}
}
