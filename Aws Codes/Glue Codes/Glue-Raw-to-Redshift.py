import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue import DynamicFrame
from pyspark.sql.functions import col, when

args = getResolvedOptions(sys.argv, ['JOB_NAME'])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

# Step 1: Read from S3 via Glue Catalog
input_frame = glueContext.create_dynamic_frame.from_catalog(
    database="default", 
    table_name="lead_scoring_small_csv",  # Assumed catalog table based on your schema; update if different
    transformation_ctx="input_frame"
)

# Step 2: Apply minimal cleaning logic inline
df = input_frame.toDF()
df = df.dropDuplicates()
for column in df.columns:
    df = df.withColumn(column, when(col(column) == 'Select', None).otherwise(col(column)))
processed_frame = DynamicFrame.fromDF(df, glueContext, "processed_frame")

# Step 3: Write to Redshift with preactions using your exact schema
glueContext.write_dynamic_frame.from_options(
    frame=processed_frame, 
    connection_type="redshift", 
    connection_options={
        "redshiftTmpDir": args.get("TempDir", "s3://******/temporary/"),
        "useConnectionProperties": "true", 
        "dbtable": "public.lead_scoring", 
        "connectionName": "redshift_connection", 
        "preactions": """
            DROP TABLE IF EXISTS public.lead_scoring;
            CREATE TABLE public.lead_scoring (
                "prospect id" VARCHAR(256),
                "lead number" BIGINT,
                "lead origin" VARCHAR(256),
                "lead source" VARCHAR(256),
                "do not email" VARCHAR(256),
                "do not call" VARCHAR(256),
                "converted" BIGINT,
                "totalvisits" BIGINT,
                "total time spent on website" BIGINT,
                "page views per visit" FLOAT,
                "last activity" VARCHAR(256),
                "country" VARCHAR(256),
                "specialization" VARCHAR(256),
                "how did you hear about x education" VARCHAR(256),
                "what is your current occupation" VARCHAR(256),
                "what matters most to you in choosing a course" VARCHAR(256),
                "search" VARCHAR(256),
                "magazine" VARCHAR(256),
                "newspaper article" VARCHAR(256),
                "x education forums" VARCHAR(256),
                "newspaper" VARCHAR(256),
                "digital advertisement" VARCHAR(256),
                "through recommendations" VARCHAR(256),
                "receive more updates about our courses" VARCHAR(256),
                "tags" VARCHAR(256),
                "lead quality" VARCHAR(256),
                "update me on supply chain content" VARCHAR(256),
                "get updates on dm content" VARCHAR(256),
                "lead profile" VARCHAR(256),
                "city" VARCHAR(256),
                "asymmetrique activity index" VARCHAR(256),
                "asymmetrique profile index" VARCHAR(256),
                "asymmetrique activity score" BIGINT,
                "asymmetrique profile score" BIGINT,
                "i agree to pay the amount through cheque" VARCHAR(256),
                "a free copy of mastering the interview" VARCHAR(256),
                "last notable activity" VARCHAR(256)
            );
        """
    }, 
    transformation_ctx="redshift_write"
)

job.commit()
