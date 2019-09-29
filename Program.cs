using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using Microsoft.ML;
using Microsoft.ML.Data;
using mlpcaptest48;
using PacketDotNet;

using SharpPcap;
using SharpPcap.LibPcap;

namespace mlpcaptest48
{
    public static class Extension
    {
        public static PacketData ToPacketData(this ParsedPacket packet)
        {
            return new PacketData
            {
                PacketBinaryData = packet.Payload,
                SrcPort = packet.SrcPort,
                DstPort = packet.DstPort,
                HeaderSize = packet.HeaderSize,
                TotalPacketLength = packet.TotalPacketSize
            };
        }
    }

    public class ParsedPacket
    {
        public bool IsTCP  { get; }

        public int SrcPort { get; }

        public int DstPort { get; }

        public string Payload { get; }

        public float HeaderSize { get; }

        public float TotalPacketSize { get; }

        public ParsedPacket(IPPacket packet, int srcPort, int dstPort, string payloadData)
        {
            IsTCP = packet.Protocol == ProtocolType.Tcp;
            HeaderSize = packet.HeaderLength;
            TotalPacketSize = packet.TotalPacketLength;

            SrcPort = srcPort;
            DstPort = dstPort;
            Payload = payloadData.Replace("-", string.Empty);
        }
        
        public ParsedPacket() { }
    }

    public class PacketData
    {
        [LoadColumn(0)]
        public bool IsTCP { get; set; }

        [LoadColumn(1)]
        public float SrcPort { get; set; }

        [LoadColumn(2)]
        public float DstPort { get; set; }

        [LoadColumn(3)]
        public float HeaderSize { get; set; }

        [LoadColumn(4)]
        public float TotalPacketLength { get; set; }

        [LoadColumn(5)]
        public string PacketBinaryData { get; set; }
    }

    public class PredictionPacketData
    {
        [ColumnName("PredictedLabel")]
        public bool IsTCP { get; set; }

        public float Probability { get; set; }

        public float Score { get; set; }
    }


    class Program
    {
        private static int packetIndex = 0;

        private static List<ParsedPacket> Packets = new List<ParsedPacket>();

        private static void TrainModel()
        {
            var mlContext = new MLContext(2019);

            using (var oFile = new System.IO.StreamWriter("feature.csv"))
            {
                foreach (var packet in Packets)
                {
                    oFile.WriteLine($"{packet.IsTCP}\t{packet.SrcPort}\t{packet.DstPort}\t{packet.HeaderSize}\t{packet.TotalPacketSize}\t{packet.Payload}");
                }
            }

            var trainingDataView = mlContext.Data.LoadFromTextFile<mlpcaptest48.PacketData>("feature.csv");

            var dataSplit = mlContext.Data.TrainTestSplit(trainingDataView, testFraction: 0.2);

            var dataProcessPipeline = mlContext.Transforms.CopyColumns("Label", nameof(PacketData.IsTCP))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("DataEncoded", "PacketBinaryData"))
                .Append(mlContext.Transforms.NormalizeMeanVariance("SrcPort"))
                .Append(mlContext.Transforms.NormalizeMeanVariance("DstPort"))
                .Append(mlContext.Transforms.NormalizeMeanVariance("HeaderSize"))
                .Append(mlContext.Transforms.Concatenate("Features", "DataEncoded", "SrcPort", "DstPort", "HeaderSize"));

            var sdcaRegressionTrainer = mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(
                labelColumnName: nameof(PacketData.IsTCP),
                featureColumnName: "Features");

            var trainingPipeline = dataProcessPipeline.Append(sdcaRegressionTrainer);

            ITransformer trainedModel = trainingPipeline.Fit(dataSplit.TrainSet);
            mlContext.Model.Save(trainedModel, dataSplit.TrainSet.Schema, "model.mdl");

            var testSetTransform = trainedModel.Transform(dataSplit.TestSet);

            var modelMetrics = mlContext.BinaryClassification.Evaluate(
                data: testSetTransform,
                labelColumnName: nameof(PacketData.IsTCP));

            Console.WriteLine(
                              $"Accuracy: {modelMetrics.Accuracy:P2}{Environment.NewLine}" +
                              $"F1Score: {modelMetrics.F1Score:P2}{Environment.NewLine}" +
                              $"Positive Recall: {modelMetrics.PositiveRecall:#.##}{Environment.NewLine}" +
                              $"Negative Recall: {modelMetrics.NegativeRecall:#.##}{Environment.NewLine}");
    }

        private static void Predict()
        {
            var mlContext = new MLContext(2019);

            var model = mlContext.Model.Load("model.mdl", out var schema);

            var predictor = mlContext.Model.CreatePredictionEngine<PacketData, PredictionPacketData>(model);
            
            var numCorrect = 0;
            var numTotal = Packets.Count;

            foreach (var packet in Packets)
            {
                var result = predictor.Predict(packet.ToPacketData());

                
                if (result.IsTCP == packet.IsTCP)
                {
                    numCorrect++;
                }
                else
                {
                    Console.WriteLine($"Prediction: {(result.IsTCP ? "TCP" : "UDP")} - Actual: {(packet.IsTCP ? "TCP" : "UDP")}");

                }

                numTotal++;
            }

            Console.WriteLine($"Efficacy: {(double)numCorrect/numTotal}");
        }

        static void Main(string[] args)
        {
            try
            {
                ICaptureDevice device;

                try
                {
                    device = new CaptureFileReaderDevice(args[0]);

                    device.Open();
                }
                catch (Exception e)
                {
                    Console.WriteLine("Caught exception when opening file" + e);
                    return;
                }

                device.OnPacketArrival += device_OnPacketArrival;

                device.Capture();

                device.Close();

                switch (args[1])
                {
                    case "train":
                        TrainModel();

                        break;
                    case "predict":
                        Predict();
                        break;
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex);
            }
        }

        private static void device_OnPacketArrival(object sender, CaptureEventArgs e)
        {
            if (e.Packet.LinkLayerType == PacketDotNet.LinkLayers.Ethernet)
            {
                var packet = PacketDotNet.Packet.ParsePacket(e.Packet.LinkLayerType, e.Packet.Data);
                var etherpacket = (EthernetPacket) packet;

                var data = BitConverter.ToString(packet.HeaderData);

                switch (etherpacket.Type)
                {
                    case EthernetType.IPv4:
                        var ippacket = (IPv4Packet) packet.PayloadPacket;

                        var srcPort = 0;
                        var dstPort = 0;

                        switch (ippacket.Protocol)
                        {
                            case ProtocolType.Udp:

                                var udpPacket = (UdpPacket)ippacket.PayloadPacket;

                                srcPort = udpPacket.SourcePort;
                                dstPort = udpPacket.DestinationPort;

                                break;
                            case ProtocolType.Tcp:
                                var tcpPacket = (TcpPacket) ippacket.PayloadPacket;

                                srcPort = tcpPacket.SourcePort;
                                dstPort = tcpPacket.DestinationPort;

                                break;
                        }
                        
                        Packets.Add(new ParsedPacket(ippacket, srcPort, dstPort, data));

                      //  Console.WriteLine($"{ippacket.Protocol} - {srcPort} - {dstPort} - {data}");

                        break;
                }
            }
        }
    }
}