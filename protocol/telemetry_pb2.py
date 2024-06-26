# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: telemetry.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


import protocol.aquatroll_pb2 as aquatroll__pb2
import protocol.message_formats_pb2 as message__formats__pb2
import protocol.mission_planning_pb2 as mission__planning__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x0ftelemetry.proto\x12\x0f\x62lueye.protocol\x1a\x0f\x61quatroll.proto\x1a\x15message_formats.proto\x1a\x16mission_planning.proto\":\n\x0b\x41ttitudeTel\x12+\n\x08\x61ttitude\x18\x01 \x01(\x0b\x32\x19.blueye.protocol.Attitude\":\n\x0b\x41ltitudeTel\x12+\n\x08\x61ltitude\x18\x01 \x01(\x0b\x32\x19.blueye.protocol.Altitude\"P\n\x12\x46orwardDistanceTel\x12:\n\x10\x66orward_distance\x18\x01 \x01(\x0b\x32 .blueye.protocol.ForwardDistance\"S\n\x13PositionEstimateTel\x12<\n\x11position_estimate\x18\x01 \x01(\x0b\x32!.blueye.protocol.PositionEstimate\"1\n\x08\x44\x65pthTel\x12%\n\x05\x64\x65pth\x18\x01 \x01(\x0b\x32\x16.blueye.protocol.Depth\"=\n\x0cReferenceTel\x12-\n\treference\x18\x01 \x01(\x0b\x32\x1a.blueye.protocol.Reference\"Z\n\x15ReferenceAutoPilotTel\x12\x41\n\x14reference_auto_pilot\x18\x01 \x01(\x0b\x32#.blueye.protocol.ReferenceAutoPilot\"J\n\x10MissionStatusTel\x12\x36\n\x0emission_status\x18\x01 \x01(\x0b\x32\x1e.blueye.protocol.MissionStatus\"F\n\x0fNotificationTel\x12\x33\n\x0cnotification\x18\x01 \x01(\x0b\x32\x1d.blueye.protocol.Notification\"G\n\x0f\x43ontrolForceTel\x12\x34\n\rcontrol_force\x18\x01 \x01(\x0b\x32\x1d.blueye.protocol.ControlForce\"S\n\x13\x43ontrollerHealthTel\x12<\n\x11\x63ontroller_health\x18\x01 \x01(\x0b\x32!.blueye.protocol.ControllerHealth\"4\n\tLightsTel\x12\'\n\x06lights\x18\x01 \x01(\x0b\x32\x17.blueye.protocol.Lights\"=\n\x12GuestPortLightsTel\x12\'\n\x06lights\x18\x01 \x01(\x0b\x32\x17.blueye.protocol.Lights\"1\n\x08LaserTel\x12%\n\x05laser\x18\x01 \x01(\x0b\x32\x16.blueye.protocol.Laser\"I\n\x13PilotGPSPositionTel\x12\x32\n\x08position\x18\x01 \x01(\x0b\x32 .blueye.protocol.LatLongPosition\"D\n\x0eRecordStateTel\x12\x32\n\x0crecord_state\x18\x01 \x01(\x0b\x32\x1c.blueye.protocol.RecordState\"N\n\x11TimeLapseStateTel\x12\x39\n\x10time_lapse_state\x18\x01 \x01(\x0b\x32\x1f.blueye.protocol.TimeLapseState\"7\n\nBatteryTel\x12)\n\x07\x62\x61ttery\x18\x01 \x01(\x0b\x32\x18.blueye.protocol.Battery\"E\n\x11\x42\x61tteryBQ40Z50Tel\x12\x30\n\x07\x62\x61ttery\x18\x01 \x01(\x0b\x32\x1f.blueye.protocol.BatteryBQ40Z50\";\n\x0b\x44iveTimeTel\x12,\n\tdive_time\x18\x01 \x01(\x0b\x32\x19.blueye.protocol.DiveTime\"z\n\x0c\x44roneTimeTel\x12\x34\n\x0freal_time_clock\x18\x01 \x01(\x0b\x32\x1b.blueye.protocol.SystemTime\x12\x34\n\x0fmonotonic_clock\x18\x02 \x01(\x0b\x32\x1b.blueye.protocol.SystemTime\"M\n\x13WaterTemperatureTel\x12\x36\n\x0btemperature\x18\x01 \x01(\x0b\x32!.blueye.protocol.WaterTemperature\"I\n\x11\x43PUTemperatureTel\x12\x34\n\x0btemperature\x18\x01 \x01(\x0b\x32\x1f.blueye.protocol.CPUTemperature\"V\n\x19\x43\x61nisterTopTemperatureTel\x12\x39\n\x0btemperature\x18\x01 \x01(\x0b\x32$.blueye.protocol.CanisterTemperature\"Y\n\x1c\x43\x61nisterBottomTemperatureTel\x12\x39\n\x0btemperature\x18\x01 \x01(\x0b\x32$.blueye.protocol.CanisterTemperature\"M\n\x16\x43\x61nisterTopHumidityTel\x12\x33\n\x08humidity\x18\x01 \x01(\x0b\x32!.blueye.protocol.CanisterHumidity\"P\n\x19\x43\x61nisterBottomHumidityTel\x12\x33\n\x08humidity\x18\x01 \x01(\x0b\x32!.blueye.protocol.CanisterHumidity\"L\n\x14VideoStorageSpaceTel\x12\x34\n\rstorage_space\x18\x01 \x01(\x0b\x32\x1d.blueye.protocol.StorageSpace\"K\n\x13\x44\x61taStorageSpaceTel\x12\x34\n\rstorage_space\x18\x01 \x01(\x0b\x32\x1d.blueye.protocol.StorageSpace\"S\n\x13\x43\x61librationStateTel\x12<\n\x11\x63\x61libration_state\x18\x01 \x01(\x0b\x32!.blueye.protocol.CalibrationState\"N\n\x14TiltStabilizationTel\x12\x36\n\x05state\x18\x01 \x01(\x0b\x32\'.blueye.protocol.TiltStabilizationState\"8\n\x08IperfTel\x12,\n\x06status\x18\x01 \x01(\x0b\x32\x1c.blueye.protocol.IperfStatus\"A\n\rNStreamersTel\x12\x30\n\x0bn_streamers\x18\x01 \x01(\x0b\x32\x1b.blueye.protocol.NStreamers\"9\n\x0cTiltAngleTel\x12)\n\x05\x61ngle\x18\x01 \x01(\x0b\x32\x1a.blueye.protocol.TiltAngle\">\n\x0c\x44roneInfoTel\x12.\n\ndrone_info\x18\x01 \x01(\x0b\x32\x1a.blueye.protocol.DroneInfo\"A\n\rErrorFlagsTel\x12\x30\n\x0b\x65rror_flags\x18\x01 \x01(\x0b\x32\x1b.blueye.protocol.ErrorFlags\"=\n\x0e\x43ontrolModeTel\x12+\n\x05state\x18\x01 \x01(\x0b\x32\x1c.blueye.protocol.ControlMode\"M\n\x11ThicknessGaugeTel\x12\x38\n\x0fthickness_gauge\x18\x01 \x01(\x0b\x32\x1f.blueye.protocol.ThicknessGauge\"8\n\nCpProbeTel\x12*\n\x08\x63p_probe\x18\x01 \x01(\x0b\x32\x18.blueye.protocol.CpProbe\"S\n\x19\x41quaTrollProbeMetadataTel\x12\x36\n\x05probe\x18\x01 \x01(\x0b\x32\'.blueye.protocol.AquaTrollProbeMetadata\"\\\n\x1a\x41quaTrollSensorMetadataTel\x12>\n\x07sensors\x18\x01 \x01(\x0b\x32-.blueye.protocol.AquaTrollSensorMetadataArray\"`\n\x1c\x41quaTrollSensorParametersTel\x12@\n\x07sensors\x18\x01 \x01(\x0b\x32/.blueye.protocol.AquaTrollSensorParametersArray\"p\n\x13\x43onnectedClientsTel\x12\x1c\n\x14\x63lient_id_in_control\x18\x01 \x01(\r\x12;\n\x11\x63onnected_clients\x18\x02 \x03(\x0b\x32 .blueye.protocol.ConnectedClient\"?\n\x0fGenericServoTel\x12,\n\x05servo\x18\x01 \x01(\x0b\x32\x1d.blueye.protocol.GenericServo\"C\n\x11MultibeamServoTel\x12.\n\x05servo\x18\x01 \x01(\x0b\x32\x1f.blueye.protocol.MultibeamServo\"I\n\x13GuestPortCurrentTel\x12\x32\n\x07\x63urrent\x18\x01 \x01(\x0b\x32!.blueye.protocol.GuestPortCurrent\"5\n\x10\x43\x61libratedImuTel\x12!\n\x03imu\x18\x01 \x01(\x0b\x32\x14.blueye.protocol.Imu\",\n\x07Imu1Tel\x12!\n\x03imu\x18\x01 \x01(\x0b\x32\x14.blueye.protocol.Imu\",\n\x07Imu2Tel\x12!\n\x03imu\x18\x01 \x01(\x0b\x32\x14.blueye.protocol.Imu\"R\n\x19MedusaSpectrometerDataTel\x12\x35\n\x04\x64\x61ta\x18\x01 \x01(\x0b\x32\'.blueye.protocol.MedusaSpectrometerData\":\n\rOculusPingTel\x12)\n\x04ping\x18\x01 \x01(\x0b\x32\x1b.blueye.protocol.OculusPing\"@\n\x0fOculusConfigTel\x12-\n\x06\x63onfig\x18\x01 \x01(\x0b\x32\x1d.blueye.protocol.OculusConfig\"I\n\x12OculusDiscoveryTel\x12\x33\n\tdiscovery\x18\x01 \x01(\x0b\x32 .blueye.protocol.OculusDiscoveryB\x1b\xaa\x02\x18\x42lueye.Protocol.Protobufb\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'telemetry_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'\252\002\030Blueye.Protocol.Protobuf'
  _globals['_ATTITUDETEL']._serialized_start=100
  _globals['_ATTITUDETEL']._serialized_end=158
  _globals['_ALTITUDETEL']._serialized_start=160
  _globals['_ALTITUDETEL']._serialized_end=218
  _globals['_FORWARDDISTANCETEL']._serialized_start=220
  _globals['_FORWARDDISTANCETEL']._serialized_end=300
  _globals['_POSITIONESTIMATETEL']._serialized_start=302
  _globals['_POSITIONESTIMATETEL']._serialized_end=385
  _globals['_DEPTHTEL']._serialized_start=387
  _globals['_DEPTHTEL']._serialized_end=436
  _globals['_REFERENCETEL']._serialized_start=438
  _globals['_REFERENCETEL']._serialized_end=499
  _globals['_REFERENCEAUTOPILOTTEL']._serialized_start=501
  _globals['_REFERENCEAUTOPILOTTEL']._serialized_end=591
  _globals['_MISSIONSTATUSTEL']._serialized_start=593
  _globals['_MISSIONSTATUSTEL']._serialized_end=667
  _globals['_NOTIFICATIONTEL']._serialized_start=669
  _globals['_NOTIFICATIONTEL']._serialized_end=739
  _globals['_CONTROLFORCETEL']._serialized_start=741
  _globals['_CONTROLFORCETEL']._serialized_end=812
  _globals['_CONTROLLERHEALTHTEL']._serialized_start=814
  _globals['_CONTROLLERHEALTHTEL']._serialized_end=897
  _globals['_LIGHTSTEL']._serialized_start=899
  _globals['_LIGHTSTEL']._serialized_end=951
  _globals['_GUESTPORTLIGHTSTEL']._serialized_start=953
  _globals['_GUESTPORTLIGHTSTEL']._serialized_end=1014
  _globals['_LASERTEL']._serialized_start=1016
  _globals['_LASERTEL']._serialized_end=1065
  _globals['_PILOTGPSPOSITIONTEL']._serialized_start=1067
  _globals['_PILOTGPSPOSITIONTEL']._serialized_end=1140
  _globals['_RECORDSTATETEL']._serialized_start=1142
  _globals['_RECORDSTATETEL']._serialized_end=1210
  _globals['_TIMELAPSESTATETEL']._serialized_start=1212
  _globals['_TIMELAPSESTATETEL']._serialized_end=1290
  _globals['_BATTERYTEL']._serialized_start=1292
  _globals['_BATTERYTEL']._serialized_end=1347
  _globals['_BATTERYBQ40Z50TEL']._serialized_start=1349
  _globals['_BATTERYBQ40Z50TEL']._serialized_end=1418
  _globals['_DIVETIMETEL']._serialized_start=1420
  _globals['_DIVETIMETEL']._serialized_end=1479
  _globals['_DRONETIMETEL']._serialized_start=1481
  _globals['_DRONETIMETEL']._serialized_end=1603
  _globals['_WATERTEMPERATURETEL']._serialized_start=1605
  _globals['_WATERTEMPERATURETEL']._serialized_end=1682
  _globals['_CPUTEMPERATURETEL']._serialized_start=1684
  _globals['_CPUTEMPERATURETEL']._serialized_end=1757
  _globals['_CANISTERTOPTEMPERATURETEL']._serialized_start=1759
  _globals['_CANISTERTOPTEMPERATURETEL']._serialized_end=1845
  _globals['_CANISTERBOTTOMTEMPERATURETEL']._serialized_start=1847
  _globals['_CANISTERBOTTOMTEMPERATURETEL']._serialized_end=1936
  _globals['_CANISTERTOPHUMIDITYTEL']._serialized_start=1938
  _globals['_CANISTERTOPHUMIDITYTEL']._serialized_end=2015
  _globals['_CANISTERBOTTOMHUMIDITYTEL']._serialized_start=2017
  _globals['_CANISTERBOTTOMHUMIDITYTEL']._serialized_end=2097
  _globals['_VIDEOSTORAGESPACETEL']._serialized_start=2099
  _globals['_VIDEOSTORAGESPACETEL']._serialized_end=2175
  _globals['_DATASTORAGESPACETEL']._serialized_start=2177
  _globals['_DATASTORAGESPACETEL']._serialized_end=2252
  _globals['_CALIBRATIONSTATETEL']._serialized_start=2254
  _globals['_CALIBRATIONSTATETEL']._serialized_end=2337
  _globals['_TILTSTABILIZATIONTEL']._serialized_start=2339
  _globals['_TILTSTABILIZATIONTEL']._serialized_end=2417
  _globals['_IPERFTEL']._serialized_start=2419
  _globals['_IPERFTEL']._serialized_end=2475
  _globals['_NSTREAMERSTEL']._serialized_start=2477
  _globals['_NSTREAMERSTEL']._serialized_end=2542
  _globals['_TILTANGLETEL']._serialized_start=2544
  _globals['_TILTANGLETEL']._serialized_end=2601
  _globals['_DRONEINFOTEL']._serialized_start=2603
  _globals['_DRONEINFOTEL']._serialized_end=2665
  _globals['_ERRORFLAGSTEL']._serialized_start=2667
  _globals['_ERRORFLAGSTEL']._serialized_end=2732
  _globals['_CONTROLMODETEL']._serialized_start=2734
  _globals['_CONTROLMODETEL']._serialized_end=2795
  _globals['_THICKNESSGAUGETEL']._serialized_start=2797
  _globals['_THICKNESSGAUGETEL']._serialized_end=2874
  _globals['_CPPROBETEL']._serialized_start=2876
  _globals['_CPPROBETEL']._serialized_end=2932
  _globals['_AQUATROLLPROBEMETADATATEL']._serialized_start=2934
  _globals['_AQUATROLLPROBEMETADATATEL']._serialized_end=3017
  _globals['_AQUATROLLSENSORMETADATATEL']._serialized_start=3019
  _globals['_AQUATROLLSENSORMETADATATEL']._serialized_end=3111
  _globals['_AQUATROLLSENSORPARAMETERSTEL']._serialized_start=3113
  _globals['_AQUATROLLSENSORPARAMETERSTEL']._serialized_end=3209
  _globals['_CONNECTEDCLIENTSTEL']._serialized_start=3211
  _globals['_CONNECTEDCLIENTSTEL']._serialized_end=3323
  _globals['_GENERICSERVOTEL']._serialized_start=3325
  _globals['_GENERICSERVOTEL']._serialized_end=3388
  _globals['_MULTIBEAMSERVOTEL']._serialized_start=3390
  _globals['_MULTIBEAMSERVOTEL']._serialized_end=3457
  _globals['_GUESTPORTCURRENTTEL']._serialized_start=3459
  _globals['_GUESTPORTCURRENTTEL']._serialized_end=3532
  _globals['_CALIBRATEDIMUTEL']._serialized_start=3534
  _globals['_CALIBRATEDIMUTEL']._serialized_end=3587
  _globals['_IMU1TEL']._serialized_start=3589
  _globals['_IMU1TEL']._serialized_end=3633
  _globals['_IMU2TEL']._serialized_start=3635
  _globals['_IMU2TEL']._serialized_end=3679
  _globals['_MEDUSASPECTROMETERDATATEL']._serialized_start=3681
  _globals['_MEDUSASPECTROMETERDATATEL']._serialized_end=3763
  _globals['_OCULUSPINGTEL']._serialized_start=3765
  _globals['_OCULUSPINGTEL']._serialized_end=3823
  _globals['_OCULUSCONFIGTEL']._serialized_start=3825
  _globals['_OCULUSCONFIGTEL']._serialized_end=3889
  _globals['_OCULUSDISCOVERYTEL']._serialized_start=3891
  _globals['_OCULUSDISCOVERYTEL']._serialized_end=3964
# @@protoc_insertion_point(module_scope)
