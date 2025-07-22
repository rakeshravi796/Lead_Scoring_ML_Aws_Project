"""
    This GLue function does feature Engineering by extracting data from Redhsift and puts into a bucket for training data. 
"""


import sys
from awsglue.transforms import Transform
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue import DynamicFrame
from pyspark.sql.functions import col, when, lit, count, mean
from pyspark.ml.feature import StringIndexer, OneHotEncoder

class LeadScoringFeatureEngineerTransform(Transform):
    def __call__(self, frame, target_col="converted"):
        df = frame.toDF()
        
        # Sanitize column names
        for old_col in df.columns:
            new_col = old_col.replace(" ", "_").lower()
            if old_col != new_col:
                df = df.withColumnRenamed(old_col, new_col)
        
        # Define columns (sanitized)
        self.ohe_cols = [
            'do_not_email', 'a_free_copy_of_mastering_the_interview',
            'search', 'newspaper_article', 'x_education_forums',
            'newspaper', 'digital_advertisement', 'through_recommendations',
            "what_is_your_current_occupation"
        ]
        self.target_encoder_cols = ['lead_source', 'lead_origin', 'last_activity', 'specialization']
        self.drop_cols = [
            'prospect_id', 'lead_number', 'how_did_you_hear_about_x_education', 'lead_profile',
            'lead_quality', 'asymmetrique_profile_score', 'asymmetrique_activity_score',
            'asymmetrique_activity_index', 'asymmetrique_profile_index', 'tags',
            'last_notable_activity', 'city', 'country', 'what_matters_most_to_you_in_choosing_a_course',
            'magazine', 'receive_more_updates_about_our_courses', 'update_me_on_supply_chain_content',
            'get_updates_on_dm_content', 'i_agree_to_pay_the_amount_through_cheque','do_not_call'
        ]
        
        # Drop duplicates and replace 'Select' with NULL
        df = df.dropDuplicates()
        for column in df.columns:
            df = df.withColumn(column, when(col(column) == 'Select', lit(None)).otherwise(col(column)))
        
        # Mapping for lead source and lead origin
        lead_source_mapping = {
            'google': 'Google', 'bing': 'Search Engine', 'Google': 'Search Engine',
            'Organic Search': 'Search Engine', 'Click2call': 'Other', 'Live Chat': 'Other',
            'Social Media': 'Other', 'Press_Release': 'Other', 'Pay per Click Ads': 'Other',
            'blog': 'Other', 'WeLearn': 'Other', 'welearnblog_Home': 'Other',
            'youtubechannel': 'Other', 'testone': 'Other', 'NC_EDM': 'Other'
        }
        lead_origin_mapping = {'Lead Import': 'Other', 'Quick Add Form': 'Other'}

        for k, v in lead_source_mapping.items():
            df = df.withColumn("lead_source", when(col("lead_source") == k, v).otherwise(col("lead_source")))
        df = df.withColumn("lead_source", when(col("lead_source").isNull(), 'Other').otherwise(col("lead_source")))

        for k, v in lead_origin_mapping.items():
            df = df.withColumn("lead_origin", when(col("lead_origin") == k, v).otherwise(col("lead_origin")))
        df = df.withColumn("lead_origin", when(col("lead_origin").isNull(), 'Other').otherwise(col("lead_origin")))

        # Handle rare last activity
        last_activity_counts = df.groupBy("last_activity").agg(count("*").alias("count"))
        rare_last_activity = last_activity_counts.filter(col("count") < 10).select("last_activity").rdd.flatMap(lambda x: x).collect()
        for rare in rare_last_activity:
            df = df.withColumn("last_activity", when(col("last_activity") == rare, 'Others').otherwise(col("last_activity")))

        # Null filling
        df = df.withColumn("specialization", when(col("specialization").isNull(), 'Others').otherwise(col("specialization")))
        df = df.withColumn("what_is_your_current_occupation", when(col("what_is_your_current_occupation").isNull(), 'Unknown').otherwise(col("what_is_your_current_occupation")))
        df = df.withColumn("last_activity", when(col("last_activity").isNull(), 'Email Opened').otherwise(col("last_activity")))

        median_total_visits = df.approxQuantile("totalvisits", [0.5], 0.25)[0]
        median_page_views = df.approxQuantile("page_views_per_visit", [0.5], 0.25)[0]
        median_time_website = df.approxQuantile("total_time_spent_on_website", [0.5], 0.25)[0]

        df = df.withColumn("totalvisits", when(col("totalvisits").isNull(), median_total_visits).otherwise(col("totalvisits")))
        df = df.withColumn("page_views_per_visit", when(col("page_views_per_visit").isNull(), median_page_views).otherwise(col("page_views_per_visit")))
        df = df.withColumn("total_time_spent_on_website", when(col("total_time_spent_on_website").isNull(), median_time_website).otherwise(col("total_time_spent_on_website")))

        # Target Encoding (using mean of target_col per category)
        for col_name in self.target_encoder_cols:
            if col_name in df.columns:
                mean_df = df.groupBy(col_name).agg(mean(target_col).alias("mean_converted"))
                df = df.join(mean_df, on=col_name, how="left")
                df = df.withColumnRenamed("mean_converted", col_name + "_encoded")
                df = df.drop(col_name)

        # One-Hot Encoding (with checks)
        for col_name in self.ohe_cols:
            if col_name in df.columns:
                distinct_count = df.select(col_name).distinct().count()
                if distinct_count >= 2:
                    indexer = StringIndexer(inputCol=col_name, outputCol=col_name + "_index", handleInvalid="skip")
                    ohe = OneHotEncoder(inputCol=col_name + "_index", outputCol=col_name + "_ohe", dropLast=True)
                    df = indexer.fit(df).transform(df)
                    df = ohe.fit(df).transform(df)
                    df = df.drop(col_name, col_name + "_index")

        # Drop columns
        df = df.drop(*self.drop_cols)

        return DynamicFrame.fromDF(df, self.glue_ctx, self.name)

args = getResolvedOptions(sys.argv, ['JOB_NAME'])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

# Step 1: Read from Redshift
input_frame = glueContext.create_dynamic_frame.from_options(
    connection_type="redshift", 
    connection_options={
        "redshiftTmpDir": "s3://******/temporary/", 
        "useConnectionProperties": "true", 
        "dbtable": "public.lead_scoring", 
        "connectionName": "redshift_connection"
    }, 
    transformation_ctx="Redshift_input"
)

# Step 2: Apply custom transformer
transformer = LeadScoringFeatureEngineerTransform(glue_ctx=glueContext, name="transformed_frame")
processed_frame = transformer(input_frame, target_col="converted")  # Specify target column if different

# Step 3: Write to S3 (from your visual code)
AmazonS3_node1752832855228 = glueContext.getSink(
    path="s3://******/testing_source/", 
    connection_type="s3", 
    updateBehavior="UPDATE_IN_DATABASE", 
    partitionKeys=[], 
    enableUpdateCatalog=True, 
    transformation_ctx="AmazonS3_node1752832855228"
)
AmazonS3_node1752832855228.setCatalogInfo(catalogDatabase="default", catalogTableName="cleaned-data-for-model")
AmazonS3_node1752832855228.setFormat("csv")
AmazonS3_node1752832855228.writeFrame(processed_frame)

job.commit()
