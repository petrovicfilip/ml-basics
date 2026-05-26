# from models import MLP_SIGRegWithCenterLoss
from models import SNARIMAXAnomalyDetector
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.table import (
    TableDescriptor,
    Schema,
    DataTypes,
    FormatDescriptor,
    StreamTableEnvironment,
)
from pyflink.common.typeinfo import Types

if __name__ == "__main__":
    env = StreamExecutionEnvironment.get_execution_environment()
    env.add_jars(
        "file:///C:/flink_lib/flink-sql-connector-kafka-4.0.1-2.0.jar"
    )

    env.set_parallelism(1)
    t_env = StreamTableEnvironment.create(env)
    # env.enable_checkpointing(3600000)
    t_env.create_temporary_table(
        "kafka_source",
        TableDescriptor.for_connector("kafka")
        .schema(
            Schema.new_builder()
            .column("Dst Port", DataTypes.FLOAT())
            .column("Protocol", DataTypes.FLOAT())
            .column("Timestamp", DataTypes.STRING())
            .column("Flow Duration", DataTypes.FLOAT())
            .column("Tot Fwd Pkts", DataTypes.FLOAT())
            .column("Tot Bwd Pkts", DataTypes.FLOAT())
            .column("TotLen Fwd Pkts", DataTypes.FLOAT())
            .column("TotLen Bwd Pkts", DataTypes.FLOAT())
            .column("Fwd Pkt Len Max", DataTypes.FLOAT())
            .column("Fwd Pkt Len Min", DataTypes.FLOAT())
            .column("Fwd Pkt Len Mean", DataTypes.FLOAT())
            .column("Fwd Pkt Len Std", DataTypes.FLOAT())
            .column("Bwd Pkt Len Max", DataTypes.FLOAT())
            .column("Bwd Pkt Len Min", DataTypes.FLOAT())
            .column("Bwd Pkt Len Mean", DataTypes.FLOAT())
            .column("Bwd Pkt Len Std", DataTypes.FLOAT())
            .column("Flow Byts/s", DataTypes.FLOAT())
            .column("Flow Pkts/s", DataTypes.FLOAT())
            .column("Flow IAT Mean", DataTypes.FLOAT())
            .column("Flow IAT Std", DataTypes.FLOAT())
            .column("Flow IAT Max", DataTypes.FLOAT())
            .column("Flow IAT Min", DataTypes.FLOAT())
            .column("Fwd IAT Tot", DataTypes.FLOAT())
            .column("Fwd IAT Mean", DataTypes.FLOAT())
            .column("Fwd IAT Std", DataTypes.FLOAT())
            .column("Fwd IAT Max", DataTypes.FLOAT())
            .column("Fwd IAT Min", DataTypes.FLOAT())
            .column("Bwd IAT Tot", DataTypes.FLOAT())
            .column("Bwd IAT Mean", DataTypes.FLOAT())
            .column("Bwd IAT Std", DataTypes.FLOAT())
            .column("Bwd IAT Max", DataTypes.FLOAT())
            .column("Bwd IAT Min", DataTypes.FLOAT())
            .column("Fwd PSH Flags", DataTypes.FLOAT())
            .column("Bwd PSH Flags", DataTypes.FLOAT())
            .column("Fwd URG Flags", DataTypes.FLOAT())
            .column("Bwd URG Flags", DataTypes.FLOAT())
            .column("Fwd Header Len", DataTypes.FLOAT())
            .column("Bwd Header Len", DataTypes.FLOAT())
            .column("Fwd Pkts/s", DataTypes.FLOAT())
            .column("Bwd Pkts/s", DataTypes.FLOAT())
            .column("Pkt Len Min", DataTypes.FLOAT())
            .column("Pkt Len Max", DataTypes.FLOAT())
            .column("Pkt Len Mean", DataTypes.FLOAT())
            .column("Pkt Len Std", DataTypes.FLOAT())
            .column("Pkt Len Var", DataTypes.FLOAT())
            .column("FIN Flag Cnt", DataTypes.FLOAT())
            .column("SYN Flag Cnt", DataTypes.FLOAT())
            .column("RST Flag Cnt", DataTypes.FLOAT())
            .column("PSH Flag Cnt", DataTypes.FLOAT())
            .column("ACK Flag Cnt", DataTypes.FLOAT())
            .column("URG Flag Cnt", DataTypes.FLOAT())
            .column("CWE Flag Count", DataTypes.FLOAT())
            .column("ECE Flag Cnt", DataTypes.FLOAT())
            .column("Down/Up Ratio", DataTypes.FLOAT())
            .column("Pkt Size Avg", DataTypes.FLOAT())
            .column("Fwd Seg Size Avg", DataTypes.FLOAT())
            .column("Bwd Seg Size Avg", DataTypes.FLOAT())
            .column("Fwd Byts/b Avg", DataTypes.FLOAT())
            .column("Fwd Pkts/b Avg", DataTypes.FLOAT())
            .column("Fwd Blk Rate Avg", DataTypes.FLOAT())
            .column("Bwd Byts/b Avg", DataTypes.FLOAT())
            .column("Bwd Pkts/b Avg", DataTypes.FLOAT())
            .column("Bwd Blk Rate Avg", DataTypes.FLOAT())
            .column("Subflow Fwd Pkts", DataTypes.FLOAT())
            .column("Subflow Fwd Byts", DataTypes.FLOAT())
            .column("Subflow Bwd Pkts", DataTypes.FLOAT())
            .column("Subflow Bwd Byts", DataTypes.FLOAT())
            .column("Init Fwd Win Byts", DataTypes.FLOAT())
            .column("Init Bwd Win Byts", DataTypes.FLOAT())
            .column("Fwd Act Data Pkts", DataTypes.FLOAT())
            .column("Fwd Seg Size Min", DataTypes.FLOAT())
            .column("Active Mean", DataTypes.FLOAT())
            .column("Active Std", DataTypes.FLOAT())
            .column("Active Max", DataTypes.FLOAT())
            .column("Active Min", DataTypes.FLOAT())
            .column("Idle Mean", DataTypes.FLOAT())
            .column("Idle Std", DataTypes.FLOAT())
            .column("Idle Max", DataTypes.FLOAT())
            .column("Idle Min", DataTypes.FLOAT())
            .column("Label", DataTypes.STRING())
            .build()
        )
        .option("connector", "kafka")
        .option("topic", "ids")
        .option("properties.bootstrap.servers", "localhost:9092")
        .option("scan.startup.mode", "earliest-offset")
        .format("json")
        .build(),
    )
    input_stream = t_env.from_path("kafka_source")

    ds = t_env.to_append_stream(
        input_stream,
        Types.ROW_NAMED(
            [
                "Dst Port",
                "Protocol",
                "Timestamp",
                "Flow Duration",
                "Tot Fwd Pkts",
                "Tot Bwd Pkts",
                "TotLen Fwd Pkts",
                "TotLen Bwd Pkts",
                "Fwd Pkt Len Max",
                "Fwd Pkt Len Min",
                "Fwd Pkt Len Mean",
                "Fwd Pkt Len Std",
                "Bwd Pkt Len Max",
                "Bwd Pkt Len Min",
                "Bwd Pkt Len Mean",
                "Bwd Pkt Len Std",
                "Flow Byts/s",
                "Flow Pkts/s",
                "Flow IAT Mean",
                "Flow IAT Std",
                "Flow IAT Max",
                "Flow IAT Min",
                "Fwd IAT Tot",
                "Fwd IAT Mean",
                "Fwd IAT Std",
                "Fwd IAT Max",
                "Fwd IAT Min",
                "Bwd IAT Tot",
                "Bwd IAT Mean",
                "Bwd IAT Std",
                "Bwd IAT Max",
                "Bwd IAT Min",
                "Fwd PSH Flags",
                "Bwd PSH Flags",
                "Fwd URG Flags",
                "Bwd URG Flags",
                "Fwd Header Len",
                "Bwd Header Len",
                "Fwd Pkts/s",
                "Bwd Pkts/s",
                "Pkt Len Min",
                "Pkt Len Max",
                "Pkt Len Mean",
                "Pkt Len Std",
                "Pkt Len Var",
                "FIN Flag Cnt",
                "SYN Flag Cnt",
                "RST Flag Cnt",
                "PSH Flag Cnt",
                "ACK Flag Cnt",
                "URG Flag Cnt",
                "CWE Flag Count",
                "ECE Flag Cnt",
                "Down/Up Ratio",
                "Pkt Size Avg",
                "Fwd Seg Size Avg",
                "Bwd Seg Size Avg",
                "Fwd Byts/b Avg",
                "Fwd Pkts/b Avg",
                "Fwd Blk Rate Avg",
                "Bwd Byts/b Avg",
                "Bwd Pkts/b Avg",
                "Bwd Blk Rate Avg",
                "Subflow Fwd Pkts",
                "Subflow Fwd Byts",
                "Subflow Bwd Pkts",
                "Subflow Bwd Byts",
                "Init Fwd Win Byts",
                "Init Bwd Win Byts",
                "Fwd Act Data Pkts",
                "Fwd Seg Size Min",
                "Active Mean",
                "Active Std",
                "Active Max",
                "Active Min",
                "Idle Mean",
                "Idle Std",
                "Idle Max",
                "Idle Min",
                "Label",
            ],
            [
                Types.FLOAT(),
                Types.FLOAT(),
                Types.STRING(),
                Types.FLOAT(),
                Types.FLOAT(),
                Types.FLOAT(),
                Types.FLOAT(),
                Types.FLOAT(),
                Types.FLOAT(),
                Types.FLOAT(),
                Types.FLOAT(),
                Types.FLOAT(),
                Types.FLOAT(),
                Types.FLOAT(),
                Types.FLOAT(),
                Types.FLOAT(),
                Types.FLOAT(),
                Types.FLOAT(),
                Types.FLOAT(),
                Types.FLOAT(),
                Types.FLOAT(),
                Types.FLOAT(),
                Types.FLOAT(),
                Types.FLOAT(),
                Types.FLOAT(),
                Types.FLOAT(),
                Types.FLOAT(),
                Types.FLOAT(),
                Types.FLOAT(),
                Types.FLOAT(),
                Types.FLOAT(),
                Types.FLOAT(),
                Types.FLOAT(),
                Types.FLOAT(),
                Types.FLOAT(),
                Types.FLOAT(),
                Types.FLOAT(),
                Types.FLOAT(),
                Types.FLOAT(),
                Types.FLOAT(),
                Types.FLOAT(),
                Types.FLOAT(),
                Types.FLOAT(),
                Types.FLOAT(),
                Types.FLOAT(),
                Types.FLOAT(),
                Types.FLOAT(),
                Types.FLOAT(),
                Types.FLOAT(),
                Types.FLOAT(),
                Types.FLOAT(),
                Types.FLOAT(),
                Types.FLOAT(),
                Types.FLOAT(),
                Types.FLOAT(),
                Types.FLOAT(),
                Types.FLOAT(),
                Types.FLOAT(),
                Types.FLOAT(),
                Types.FLOAT(),
                Types.FLOAT(),
                Types.FLOAT(),
                Types.FLOAT(),
                Types.FLOAT(),
                Types.FLOAT(),
                Types.FLOAT(),
                Types.FLOAT(),
                Types.FLOAT(),
                Types.FLOAT(),
                Types.FLOAT(),
                Types.FLOAT(),
                Types.FLOAT(),
                Types.FLOAT(),
                Types.FLOAT(),
                Types.FLOAT(),
                Types.FLOAT(),
                Types.FLOAT(),
                Types.FLOAT(),
                Types.FLOAT(),
                Types.STRING(),
            ],
        ),
    )

   # ds.print()

    # stage1_results for multi-class classification
    results_ds = ds.map(
        SNARIMAXAnomalyDetector(),
        output_type=Types.ROW_NAMED(
            [
                "timestamp",
                "label",
                "flow_byts_prediction",
                "flow_pkts_prediction",
                "pkt_len_prediction",
                "flow_byts_error",
                "flow_pkts_error",
                "pkt_len_error",
                "anomaly_score"
            ],
            [
                Types.STRING(),
                Types.STRING(),
                Types.FLOAT(),
                Types.FLOAT(),
                Types.FLOAT(),
                Types.FLOAT(),
                Types.FLOAT(),
                Types.FLOAT(),
                Types.FLOAT(),
            ],
        ),
    )
    results_table = t_env.from_data_stream(results_ds)


    # results_table_schema = results_table.get_schema()

    def table_schema_to_schema(table_schema):
        """
        Creates Schema from TableSchema
        """
        builder = Schema.new_builder()
        for name, dtype in zip(
                table_schema.get_field_names(), table_schema.get_field_data_types()
        ):
            builder = builder.column(name, dtype)
        return builder


    # #binary file sink
    # t_env.create_temporary_table(
    #     "file_sink_stage",
    #     TableDescriptor.for_connector("filesystem")
    #     .schema(
    #         Schema.new_builder()
    #         .column("Timestamp", DataTypes.STRING())
    #         .column("orig_label", DataTypes.STRING())
    #         .column("binary_label", DataTypes.STRING())
    #         .column("predicted_label", DataTypes.STRING())
    #         .column("predicted_binary_label", DataTypes.STRING())
    #         .column("probability", DataTypes.FLOAT())
    #         .column("class_probabilities", DataTypes.STRING())
    #         .build()
    #     )
    #     .option("path", "./output")
    #     .format(FormatDescriptor.for_format("json").build())
    #     .build(),
    # )

    # multi-class file sink
    t_env.create_temporary_table(
        "file_sink_stage",
        TableDescriptor.for_connector("filesystem")
        .schema(
            Schema.new_builder()
            .column("timestamp", DataTypes.STRING())
            .column("label", DataTypes.STRING())

            .column("flow_byts_prediction", DataTypes.FLOAT())
            .column("flow_pkts_prediction", DataTypes.FLOAT())
            .column("pkt_len_prediction", DataTypes.FLOAT())

            .column("flow_byts_error", DataTypes.FLOAT())
            .column("flow_pkts_error", DataTypes.FLOAT())
            .column("pkt_len_error", DataTypes.FLOAT())

            .column("anomaly_score", DataTypes.FLOAT())

            .build()
        )
        .option("path", "./output")
        .format(FormatDescriptor.for_format("json").build())
        .build(),
    )

    statement_set = t_env.create_statement_set()
    statement_set.add_insert("file_sink_stage", results_table)
    statement_set.execute().wait()

    # statement_set.execute()

    # env.execute("PyFlink + River")
