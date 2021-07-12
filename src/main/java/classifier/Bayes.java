package classifier;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.LineReader;

import java.io.IOException;

public class Bayes {

    public static final String TRAIN = "/classifier/train.txt";
    public static final int dim = 20;

    public static final class TrainMapper extends Mapper<Object, Text, Text, Text> {

        private static final double[][] means = new double[2][dim];
        private static int count0 = 0;
        private static int count1 = 0;

        @Override
        protected void setup(Context context) {
            for (int i = 0; i < 2; i++) {
                for (int j = 0; j < dim; j++) {
                    means[i][j] = 0;
                }
            }
        }

        @Override
        protected void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            String line = value.toString();
            double[] point = getPoint(line);
            for (int i = 0; i < dim; i++) {
                if (point[dim] < 0.5) {
                    means[0][i] += point[i];
                } else {
                    means[1][i] += point[i];
                }
            }
            if (point[dim] < 0.5) {
                count0++;
                context.write(new Text("0"), value);
            } else if (point[dim] > 0.5) {
                count1++;
                context.write(new Text("1"), value);
            }
        }

        @Override
        protected void cleanup(Context context) throws IOException {
            StringBuilder means0 = new StringBuilder();
            StringBuilder means1 = new StringBuilder();
            for (int i = 0; i < dim; i++) {
                means[0][i] /= count0;
                means[1][i] /= count1;
                means0.append(means[0][i]);
                means1.append(means[1][i]);
                if (i < dim - 1) {
                    means0.append(",");
                    means1.append(",");
                }
            }
            outputString(means0.toString(), "/classifier/means0.txt");
            outputString(means1.toString(), "/classifier/means1.txt");
        }
    }

    public static final class TrainReducer extends Reducer<Text, Text, Text, Text> {

        private static double[][] means = new double[2][dim];
        private static double[][] sse = new double[2][dim];
        private static int count0 = 0;
        private static int count1 = 0;

        @Override
        protected void setup(Context context) throws IOException {
            Path meansPath0 = new Path("/classifier/means0.txt");
            Path meansPath1 = new Path("/classifier/means1.txt");
            Configuration conf = new Configuration();
            FileSystem fileSystem = meansPath0.getFileSystem(conf);
            FSDataInputStream inputStream0 = fileSystem.open(meansPath0);
            FSDataInputStream inputStream1 = fileSystem.open(meansPath1);
            LineReader reader0 = new LineReader(inputStream0, conf);
            LineReader reader1 = new LineReader(inputStream1, conf);
            Text line = new Text();
            String[] means0 = new String[dim];
            String[] means1 = new String[dim];
            while (reader0.readLine(line) > 0) {
                means0 = line.toString().split(",");
            }
            while (reader1.readLine(line) > 0) {
                means1 = line.toString().split(",");
            }
            for (int i = 0; i < 2; i++) {
                for (int j = 0; j < dim; j++) {
                    means[i][j] = 0;
                    sse[i][j] = 0;
                    if (i == 0) {
                        means[i][j] = Double.parseDouble(means0[j]);
                    } else {
                        means[i][j] = Double.parseDouble(means1[j]);
                    }
                }
            }
        }

        @Override
        protected void reduce(Text key, Iterable<Text> values, Context context) {
            for (Text value : values) {
                String line = value.toString();
                double[] point = getPoint(line);
                if (key.toString().equals("0")) {
                    for (int i = 0; i < dim; i++) {
                        sse[0][i] += Math.pow(point[i]-means[0][i], 2);
                        count0++;
                    }
                } else {
                    for (int i = 0; i < dim; i++) {
                        sse[1][i] += Math.pow(point[i]-means[1][i], 2);
                        count1++;
                    }
                }
            }
        }

        @Override
        protected void cleanup(Context context) throws IOException, InterruptedException {
            context.write(new Text(getString(means[0])), new Text());
            context.write(new Text(getString(means[1])), new Text());
            for (int i = 0; i < dim; i++) {
                sse[0][i] /= count0;
                sse[1][i] /= count1;
            }
            context.write(new Text(getString(sse[0])), new Text());
            context.write(new Text(getString(sse[1])), new Text());
            context.write(new Text(String.valueOf((count0+0.0) /(count0+count1))), new Text());
            context.write(new Text(String.valueOf((count1+0.0) /(count0+count1))), new Text());
        }
    }

    public static double[] getPoint(String line) {
        String[] values = line.split(",");
        double[] point = new double[dim+1];
        for (int i = 0; i < dim+1; i++) {
            point[i] = Double.parseDouble(values[i]);
        }
        return point;
    }

    public static String getString(double[] values) {
        StringBuilder builder = new StringBuilder();
        for (int i = 0; i < values.length; i++) {
            builder.append(values[i]);
            if (i < values.length-1) {
                builder.append(",");
            }
        }
        return builder.toString();
    }

    public static void outputString(String str, String file) throws IOException {
        Path filePath = new Path(file);
        Configuration conf = new Configuration();
        FileSystem fileSystem = filePath.getFileSystem(conf);
        FSDataOutputStream outputStream = fileSystem.create(filePath);
        outputStream.write(str.getBytes());
        outputStream.flush();
    }

    public static void deleteFile(String path) throws IOException {
        Configuration conf = new Configuration();
        Path filePath = new Path(path);
        FileSystem fileSystem = filePath.getFileSystem(conf);
        fileSystem.delete(filePath, true);
    }

    public static void train() throws IOException, InterruptedException, ClassNotFoundException {
        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "train");
        // jar
        job.setJarByClass(Bayes.class);
        // mapper
        job.setMapperClass(TrainMapper.class);
        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(Text.class);
        FileInputFormat.setInputPaths(job, new Path(TRAIN));
        // reducer
        job.setReducerClass(TrainReducer.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);
        FileOutputFormat.setOutputPath(job, new Path("/classifier/train"));
        // submit
        job.waitForCompletion(true);
    }

    public static void validate() throws IOException {
        Path trainPath = new Path("/classifier/train/part-r-00000");
        Path outputPath = new Path("/classifier/accuracy.txt");
        Configuration conf = new Configuration();
        FileSystem fileSystem = trainPath.getFileSystem(conf);
        FSDataInputStream inputStream = fileSystem.open(trainPath);
        FSDataOutputStream outputStream = fileSystem.create(outputPath);
        LineReader reader = new LineReader(inputStream, conf);
        Text line = new Text();

        double[][] mean = new double[2][dim];
        double[][] sigma = new double[2][dim];
        double[] priori = new double[2];
        int index = 0;
        while (reader.readLine(line) > 0) {
            switch (index) {
                case 0: {
                    String[] means0 = line.toString().split(",");
                    for (int i = 0; i < dim; i++) {
                        mean[0][i] = Double.parseDouble(means0[i]);
                    }
                    break;
                }
                case 1: {
                    String[] means1 = line.toString().split(",");
                    for (int i = 0; i < dim; i++) {
                        mean[1][i] = Double.parseDouble(means1[i]);
                    }
                    break;
                }
                case 2: {
                    String[] sigma0 = line.toString().split(",");
                    for (int i = 0; i < dim; i++) {
                        sigma[0][i] = Double.parseDouble(sigma0[i]);
                    }
                    break;
                }
                case 3: {
                    String[] sigma1 = line.toString().split(",");
                    for (int i = 0; i < dim; i++) {
                        sigma[1][i] = Double.parseDouble(sigma1[i]);
                    }
                    break;
                }
                case 4: {
                    priori[0] = Double.parseDouble(line.toString());
                    break;
                }
                case 5: {
                    priori[1] = Double.parseDouble(line.toString());
                    break;
                }
            }
            index++;
        }

        Path dataPath = new Path("/classifier/validate.txt");
        FSDataInputStream dataInputStream = fileSystem.open(dataPath);
        LineReader lineReader = new LineReader(dataInputStream, conf);
        double accuracy = 0;
        int count = 0;
        Text data = new Text();
        while (lineReader.readLine(data) > 0) {
            double[] point = getPoint(data.toString());
            int classification = classify(point, mean, sigma, priori[0], priori[1]);
            if (classification == 0 && point[dim] < 0.5) {
                accuracy++;
            } else if (classification == 1 && point[dim] > 0.5) {
                accuracy++;
            }
            count++;
        }
        String output = "accuracy: " + accuracy / count;
        outputStream.write(output.getBytes());
        outputStream.flush();
    }

    public static int classify(double[] point, double[][] mean, double[][] sigma, double p0, double p1) {
        double odd = 1;
        for (int i = 0; i < dim; i++) {
            odd = odd * Math.exp(-Math.pow((point[i]-mean[0][i]), 2)*0.5/sigma[0][i]) / Math.sqrt(2*Math.PI*sigma[0][i]);
            odd = odd / Math.exp(-Math.pow((point[i]-mean[1][i]), 2)*0.5/sigma[1][i]) * Math.sqrt(2*Math.PI*sigma[1][i]);
        }
        odd = odd * p0 / p1;
        if (odd > 1) {
            return 0;
        } else return 1;
    }

    public static void test() throws IOException {
        Path trainPath = new Path("/classifier/train/part-r-00000");
        Path outputPath = new Path("/classifier/output.txt");
        Configuration conf = new Configuration();
        FileSystem fileSystem = trainPath.getFileSystem(conf);
        FSDataInputStream inputStream = fileSystem.open(trainPath);
        FSDataOutputStream outputStream = fileSystem.create(outputPath);
        LineReader reader = new LineReader(inputStream, conf);
        Text line = new Text();
        double[][] mean = new double[2][dim];
        double[][] sigma = new double[2][dim];
        double[] priori = new double[2];
        int index = 0;
        while (reader.readLine(line) > 0) {
            switch (index) {
                case 0: {
                    String[] means0 = line.toString().split(",");
                    for (int i = 0; i < dim; i++) {
                        mean[0][i] = Double.parseDouble(means0[i]);
                    }
                    break;
                }
                case 1: {
                    String[] means1 = line.toString().split(",");
                    for (int i = 0; i < dim; i++) {
                        mean[1][i] = Double.parseDouble(means1[i]);
                    }
                    break;
                }
                case 2: {
                    String[] sigma0 = line.toString().split(",");
                    for (int i = 0; i < dim; i++) {
                        sigma[0][i] = Double.parseDouble(sigma0[i]);
                    }
                    break;
                }
                case 3: {
                    String[] sigma1 = line.toString().split(",");
                    for (int i = 0; i < dim; i++) {
                        sigma[1][i] = Double.parseDouble(sigma1[i]);
                    }
                    break;
                }
                case 4: {
                    priori[0] = Double.parseDouble(line.toString());
                    break;
                }
                case 5: {
                    priori[1] = Double.parseDouble(line.toString());
                    break;
                }
            }
            index++;
        }
        Path dataPath = new Path("/classifier/test.txt");
        FSDataInputStream dataInputStream = fileSystem.open(dataPath);
        LineReader lineReader = new LineReader(dataInputStream, conf);
        while (lineReader.readLine(line) > 0) {
            String[] values = line.toString().split(",");
            double[] point = new double[dim];
            for (int i = 0; i < dim; i++) {
                point[i] = Double.parseDouble(values[i]);
            }
            String result = classify(point, mean, sigma, priori[0], priori[1]) + "\n";
            outputStream.write(result.getBytes());
            outputStream.flush();
        }
    }

    public static void main(String[] args) throws IOException, InterruptedException, ClassNotFoundException {
        deleteFile("/classifier/means0.txt");
        deleteFile("/classifier/means1.txt");
        deleteFile("/classifier/train");
        deleteFile("/classifier/accuracy.txt");
        train();
        deleteFile("/classifier/means0.txt");
        deleteFile("/classifier/means1.txt");
        validate();
        test();
    }
}
