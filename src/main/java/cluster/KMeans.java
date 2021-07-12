package cluster;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IOUtils;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.LineReader;

import java.io.IOException;
import java.util.*;

public class KMeans {

    public static final String DATA = "/cluster/cluster.txt";
    public static final String CENTER = "/cluster/center/part-r-00000";
    public static final String NEW_CENTER = "/cluster/new_center/part-r-00000";
    public static final double delta = 1e-2;
    public static final int dim = 20;
    public static int k = 13;

    public static final class SampleMapper extends Mapper<Object, Text, Text, Text> {

        private static final List<Text> centers = new ArrayList<>();
        private int i = 0;
        private int k;

        @Override
        protected void setup(Context context) {
            k = Integer.parseInt(context.getConfiguration().get("k"));
        }

        @Override
        protected void map(Object key, Text value, Context context) {
            if (i < k) {
                centers.add(new Text(value));
            } else {
                Random random = new Random();
                int m = random.nextInt(i);
                if (m < k) {
                    centers.set(m, new Text(value));
                }
            }
            i++;
        }

        @Override
        protected void cleanup(Context context) throws IOException, InterruptedException {
            for (int i = 0; i < k; i++) {
                context.write(new Text(String.valueOf(i)), new Text(centers.get(i)));
            }
        }
    }

    public static final class SampleReducer extends Reducer<Text, Text, Text, Text> {

        @Override
        protected void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            for (Text value : values) {
                context.write(key, value);
                break;
            }
        }
    }

    public static final class KMeansMapper extends Mapper<Object, Text, Text, Text> {

        private static double[][] centers = new double[k][dim];

        @Override
        protected void setup(Context context) throws IOException {
            centers = getCenters(CENTER);
        }

        @Override
        protected void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            double[] point = getPoint(value);
            double minDistance = Double.MAX_VALUE;
            int minCluster = -1;
            for (int i = 0; i < k; i++) {
                double distance = 0;
                for (int j = 0; j < dim; j++) {
                    distance = distance + Math.pow(Math.abs(centers[i][j] - point[j]), 2);
                }
                if (distance < minDistance) {
                    minDistance = distance;
                    minCluster = i;
                }
            }
            context.write(new Text(String.valueOf(minCluster)), value);
        }
    }

    public static final class KMeansReducer extends Reducer<Text, Text, Text, Text> {
        @Override
        protected void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            List<double[]> pointList = new ArrayList<>();
            for (Text value : values) {
                pointList.add(getPoint(value));
            }
            double[] mean = mean(pointList);
            context.write(key, new Text(getText(mean)));
        }
    }

    public static void sample() throws IOException, InterruptedException, ClassNotFoundException {
        Configuration conf = new Configuration();
        conf.set("k", String.valueOf(k));
        Job job = Job.getInstance(conf, "sample");
        // jar
        job.setJarByClass(KMeans.class);
        // mapper
        job.setMapperClass(SampleMapper.class);
        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(Text.class);
        FileInputFormat.setInputPaths(job, new Path(DATA));
        // reducer
        job.setReducerClass(SampleReducer.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);
        FileOutputFormat.setOutputPath(job, new Path("/cluster/center"));
        // submit
        job.waitForCompletion(true);
    }

    public static void run() throws IOException, InterruptedException, ClassNotFoundException {
        Configuration conf = new Configuration();
        conf.set("k", String.valueOf(k));
        Job job = Job.getInstance(conf, "kMeans");
        // jar
        job.setJarByClass(KMeans.class);
        // mapper
        job.setMapperClass(KMeansMapper.class);
        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(Text.class);
        FileInputFormat.setInputPaths(job, new Path(DATA));
        // reducer
        job.setReducerClass(KMeansReducer.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);
        FileOutputFormat.setOutputPath(job, new Path("/cluster/new_center"));
        // submit
        job.waitForCompletion(true);
    }

    public static double[][] getCenters(String path) throws IOException {
        double[][] centers = new double[k][dim];
        Path centerPath = new Path(path);
        Configuration conf = new Configuration();
        FileSystem fileSystem = centerPath.getFileSystem(conf);
        FSDataInputStream inputStream = fileSystem.open(centerPath);
        LineReader reader = new LineReader(inputStream, conf);
        Text line = new Text();
        while (reader.readLine(line) > 0) {
            String[] params = line.toString().split("\\s+");
            int cluster = Integer.parseInt(params[0]);
            String attributes = params[1];
            String[] values = attributes.split(",");
            for (int i = 0; i < values.length; i++) {
                centers[cluster][i] = Double.parseDouble(values[i]);
            }
        }
        return centers;
    }

    public static double[] getPoint(Text text) {
        double[] point = new double[dim];
        String[] values = text.toString().split(",");
        for (int i = 0; i < values.length; i++) {
            point[i] = Double.parseDouble(values[i]);
        }
        return point;
    }

    public static String getText(double[] values) {
        StringBuilder builder = new StringBuilder();
        for (double value : values) {
            builder.append(value).append(",");
        }
        return builder.substring(0, builder.length()-1);
    }

    public static double[] mean(List<double[]> list) {
        double[] mean = new double[dim];
        for (int i = 0; i < dim; i++) {
            mean[i] = 0;
        }
        for (double[] point : list) {
            for (int i = 0; i < dim; i++) {
                mean[i] = mean[i] + point[i];
            }
        }
        for (int i = 0; i < dim; i++) {
            mean[i] = mean[i] / list.size();
        }
        return mean;
    }

    public static void clear() throws IOException {
        deleteFile("/cluster/center");
        deleteFile("/cluster/new_center");
        deleteFile("/cluster/output.txt");
        deleteFile("/cluster/sse" + k + ".txt");
    }

    public static void deleteFile(String path) throws IOException {
        Configuration conf = new Configuration();
        Path filePath = new Path(path);
        FileSystem fileSystem = filePath.getFileSystem(conf);
        fileSystem.delete(filePath, true);
    }

    public static boolean converge() throws IOException {
        double[][] centers = getCenters(CENTER);
        double[][] newCenters = getCenters(NEW_CENTER);
        for (int i = 0; i < k; i ++) {
            for (int j = 0; j < dim; j++) {
                if (Math.abs(centers[i][j] - newCenters[i][j]) > delta) {
                    return false;
                }
            }
        }
        return true;
    }

    public static void replaceCenters() throws IOException {
        deleteFile(CENTER);
        Configuration conf = new Configuration();
        Path destinationPath = new Path(CENTER);
        Path sourcePath = new Path(NEW_CENTER);
        FileSystem fileSystem = sourcePath.getFileSystem(conf);
        FSDataOutputStream outputStream = fileSystem.create(destinationPath, true);
        FSDataInputStream inputStream = fileSystem.open(sourcePath);
        IOUtils.copyBytes(inputStream, outputStream, 4096, true);
        deleteFile("/cluster/new_center");
    }

    public static void sse() throws IOException {
        double[][] centers = getCenters(CENTER);
        Path dataPath = new Path(DATA);
        Configuration conf = new Configuration();
        FileSystem fileSystem = dataPath.getFileSystem(conf);
        FSDataInputStream inputStream = fileSystem.open(dataPath);
        LineReader reader = new LineReader(inputStream, conf);
        FSDataOutputStream outputStream = fileSystem.create(new Path("/cluster/sse" + k + ".txt"));
        double sse = 0;
        Text line = new Text();
        while (reader.readLine(line) > 0) {
            double[] point = getPoint(line);
            double minDistance = Double.MAX_VALUE;
            for (int i = 0; i < k; i++) {
                double distance = 0;
                for (int j = 0; j < dim; j++) {
                    distance = distance + Math.pow(Math.abs(centers[i][j] - point[j]), 2);
                }
                if (distance < minDistance) {
                    minDistance = distance;
                }
            }
            sse = sse + minDistance;
        }
        String result = "k = " + k + ", sse = " + sse + "\n";
        outputStream.write(result.getBytes());
        outputStream.flush();
    }

    public static void output() throws IOException {
        double[][] centers = getCenters(CENTER);
        Path dataPath = new Path(DATA);
        Configuration conf = new Configuration();
        FileSystem fileSystem = dataPath.getFileSystem(conf);
        FSDataInputStream inputStream = fileSystem.open(dataPath);
        LineReader reader = new LineReader(inputStream, conf);
        FSDataOutputStream outputStream = fileSystem.create(new Path("/cluster/output.txt"));
        Text line = new Text();
        while (reader.readLine(line) > 0) {
            double[] point = getPoint(line);
            double minDistance = Double.MAX_VALUE;
            int minCluster = -1;
            for (int i = 0; i < k; i++) {
                double distance = 0;
                for (int j = 0; j < dim; j++) {
                    distance = distance + Math.pow(Math.abs(centers[i][j] - point[j]), 2);
                }
                if (distance < minDistance) {
                    minDistance = distance;
                    minCluster = i;
                }
            }
            String result = minCluster + "\n";
            outputStream.write(result.getBytes());
        }
        outputStream.flush();
    }

    public static void main(String[] args) throws IOException, InterruptedException, ClassNotFoundException {
        // clear centers
        clear();
        // sample
        sample();
        // kMeans
        while (true) {
            run();
            if (converge()) {
                break;
            } else {
                replaceCenters();
            }
        }
        sse();
        output();
    }
}
