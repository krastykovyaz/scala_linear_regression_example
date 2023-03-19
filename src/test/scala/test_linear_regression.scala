import breeze.linalg._
import breeze.stats.distributions._

object test_linear_regression {
  def main(args: Array[String]): Unit = {
    //standard Gaussian distribution for the means of inputs
    val mean_dist = Gaussian(0.0, 1.0)

    //lognormal distribution for the variance of inputs
    val var_dist = LogNormal(5.0, 3.0)
    //The number of observations (cases/ instances):
    val N = 100
    val max_index_train = (N / 2) - 1
    val start_index_test = max_index_train + 1

    //The number of features (inputs/ variables):
    val p = 10
    //empty data matrix, the last column is reserved for the output (y)
    val X = DenseMatrix.zeros[Double](N, p + 1)
    //The first half of the observations is stored as training set.
    val X_train = DenseMatrix.zeros[Double]((N / 2), p + 1)
    //The second half of the observations is stored as test set.
    val X_test = DenseMatrix.zeros[Double]((N / 2), p + 1)
    //create all inputs (predictors)
    for (i <- 0 to (p - 1)) {
      val mean = mean_dist.draw()
      val sd = var_dist.draw()
      val dist = Gaussian(mean, sd)
      val x = dist.sample(N)
      // matrix assignment to column
      X(::, i) := DenseVector(x.toArray)
      X_train(::, i) := DenseVector(x.toArray)(0 to max_index_train)
      X_test(::, i) := DenseVector(x.toArray)(start_index_test to (N - 1))
      // calculate y based on inputs and random coefficients
      val coefs = mean_dist.sample(p + 1)
      val c = DenseVector(coefs.toArray)
      // add dummy value 0 for y column
      c(p) = 0.0
      val y = X * c
      val y_train = y(0 to max_index_train)
      val y_test = y(start_index_test to (N - 1))
      // estimate the coefficients using least squares:
      val b = pinv(X.t * X) * X.t * y

      println("True coefficients:")
      println(c)
      println("Estimated coefficients:")
      println(b)
    }
    println("True coefficients are so closed with predicted coefficients !")
    for (i <- 0 to (p - 1)) {
      //create a vector of random numbers (uniform random numbers between 0 and 1)
      val noise = DenseVector(mean_dist.sample(N).toArray)
      val coefs = mean_dist.sample(p + 1)
      val c = DenseVector(coefs.toArray)
      c(p) = 0.0
      //add the noise vector with mean zero and variance 1
      val y_ = X * c + noise
      //estimate the coefficients using least squares:
      val b_ = pinv(X.t * X) * X.t * y_

      println("True coefficients:")
      println(c)
      println("Estimated coefficients:")
      println(b_)
    }
    println("True coefficients are so closed with predicted coefficients in noises as well !")
  }
  def test(args: Array[String]): Unit = {
    //standard Gaussian distribution for the means of inputs
    val mean_dist = Gaussian(0.0, 1.0)

    //lognormal distribution for the variance of inputs
    val var_dist = LogNormal(5.0, 3.0)
    //The number of observations (cases/ instances):
    val N = 100
    val max_index_train = (N / 2) - 1
    val start_index_test = max_index_train + 1

    //The number of features (inputs/ variables):
    val p = 10
    //empty data matrix, the last column is reserved for the output (y)
    val X = DenseMatrix.zeros[Double](N, p + 1)
    //The first half of the observations is stored as training set.
    val X_train = DenseMatrix.zeros[Double]((N / 2), p + 1)
    //The second half of the observations is stored as test set.
    val X_test = DenseMatrix.zeros[Double]((N / 2), p + 1)
    //create all inputs (predictors)
    for (i <- 0 to (p - 1)) {
      val mean = mean_dist.draw()
      val sd = var_dist.draw()
      val dist = Gaussian(mean, sd)
      val x = dist.sample(N)
      // matrix assignment to column
      X(::, i) := DenseVector(x.toArray)
      X_train(::, i) := DenseVector(x.toArray)(0 to max_index_train)
      X_test(::, i) := DenseVector(x.toArray)(start_index_test to (N - 1))
      // calculate y based on inputs and random coefficients
      val coefs = mean_dist.sample(p + 1)
      val c = DenseVector(coefs.toArray)
      // add dummy value 0 for y column
      c(p) = 0.0
      val y = X * c
      val y_train = y(0 to max_index_train)
      val y_test = y(start_index_test to (N - 1))
      // estimate the coefficients using least squares:
      val b = pinv(X.t * X) * X.t * y

      println("True coefficients:")
      println(c)
      println("Estimated coefficients:")
      println(b)
    }
    println("True coefficients are so closed with predicted coefficients !")
    for (i <- 0 to (p - 1)) {
      //create a vector of random numbers (uniform random numbers between 0 and 1)
      val noise = DenseVector(mean_dist.sample(N).toArray)
      val coefs = mean_dist.sample(p + 1)
      val c = DenseVector(coefs.toArray)
      c(p) = 0.0
      //add the noise vector with mean zero and variance 1
      val y = X * c + noise
      //estimate the coefficients using least squares:
      val b = pinv(X.t * X) * X.t * y

      println("True coefficients:")
      println(c)
      println("Estimated coefficients:")
      println(b)
    }
    println("True coefficients are so closed with predicted coefficients in noises as well !")
  }
}
