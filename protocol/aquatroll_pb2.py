# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: aquatroll.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from google.protobuf import timestamp_pb2 as google_dot_protobuf_dot_timestamp__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x0f\x61quatroll.proto\x12\x0f\x62lueye.protocol\x1a\x1fgoogle/protobuf/timestamp.proto\"\xcc\x02\n\x17\x41quaTrollParameterBlock\x12\x16\n\x0emeasured_value\x18\x01 \x01(\x02\x12\x39\n\x0cparameter_id\x18\x02 \x01(\x0e\x32#.blueye.protocol.AquaTrollParameter\x12\x30\n\x08units_id\x18\x03 \x01(\x0e\x32\x1e.blueye.protocol.AquaTrollUnit\x12;\n\x10\x64\x61ta_quality_ids\x18\x07 \x03(\x0e\x32!.blueye.protocol.AquaTrollQuality\x12\x1f\n\x17off_line_sentinel_value\x18\x05 \x01(\x02\x12\x37\n\x0f\x61vailable_units\x18\x06 \x03(\x0e\x32\x1e.blueye.protocol.AquaTrollUnitJ\x04\x08\x04\x10\x05R\x0f\x64\x61ta_quality_id\"\xef\x07\n\x17\x41quaTrollSensorMetadata\x12-\n\ttimestamp\x18\x01 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x12\x33\n\tsensor_id\x18\x02 \x01(\x0e\x32 .blueye.protocol.AquaTrollSensor\x12\x1c\n\x14sensor_serial_number\x18\x03 \x01(\r\x12\x43\n\x13sensor_status_flags\x18\x17 \x03(\x0e\x32&.blueye.protocol.AquaTrollSensorStatus\x12<\n\x18last_factory_calibration\x18\x05 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x12<\n\x18next_factory_calibration\x18\x06 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x12\x39\n\x15last_user_calibration\x18\x07 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x12\x39\n\x15next_user_calibration\x18\x08 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x12$\n\x1cwarm_up_time_in_milliseconds\x18\t \x01(\r\x12(\n fast_sample_rate_in_milliseconds\x18\n \x01(\r\x12#\n\x1bnumber_of_sensor_parameters\x18\x0b \x01(\r\x12*\n\"alarm_and_warning_parameter_number\x18\x0c \x01(\r\x12%\n\x1d\x61larm_and_warning_enable_bits\x18\r \x01(\r\x12\x1c\n\x14high_alarm_set_value\x18\x0e \x01(\x02\x12\x1e\n\x16high_alarm_clear_value\x18\x0f \x01(\x02\x12\x1e\n\x16high_warning_set_value\x18\x10 \x01(\x02\x12 \n\x18high_warning_clear_value\x18\x11 \x01(\x02\x12\x1f\n\x17low_warning_clear_value\x18\x12 \x01(\x02\x12\x1d\n\x15low_warning_set_value\x18\x13 \x01(\x02\x12\x1d\n\x15low_alarm_clear_value\x18\x14 \x01(\x02\x12\x1b\n\x13low_alarm_set_value\x18\x15 \x01(\x02\x12\x42\n\x10parameter_blocks\x18\x16 \x03(\x0b\x32(.blueye.protocol.AquaTrollParameterBlockJ\x04\x08\x04\x10\x05R\rsensor_status\"\x88\x01\n\x1c\x41quaTrollSensorMetadataArray\x12-\n\ttimestamp\x18\x01 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x12\x39\n\x07sensors\x18\x02 \x03(\x0b\x32(.blueye.protocol.AquaTrollSensorMetadata\"\xdf\x06\n\x16\x41quaTrollProbeMetadata\x12-\n\ttimestamp\x18\x01 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x12\x0e\n\x06status\x18\x18 \x01(\x08\x12%\n\x1dregister_map_template_version\x18\x02 \x01(\r\x12\x33\n\tdevice_id\x18\x03 \x01(\x0e\x32 .blueye.protocol.AquaTrollDevice\x12\x1c\n\x14\x64\x65vice_serial_number\x18\x04 \x01(\r\x12\x34\n\x10manufacture_date\x18\x05 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x12\x18\n\x10\x66irmware_version\x18\x06 \x01(\r\x12\x19\n\x11\x62oot_code_version\x18\x07 \x01(\r\x12\x18\n\x10hardware_version\x18\x08 \x01(\r\x12\x15\n\rmax_data_logs\x18\t \x01(\r\x12\x1d\n\x15total_data_log_memory\x18\n \x01(\r\x12\x1b\n\x13total_battery_ticks\x18\x0b \x01(\r\x12\x37\n\x13last_battery_change\x18\x0c \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x12\x13\n\x0b\x64\x65vice_name\x18\r \x01(\t\x12\x11\n\tsite_name\x18\x0e \x01(\t\x12\x1b\n\x13latitude_coordinate\x18\x0f \x01(\x01\x12\x1c\n\x14longitude_coordinate\x18\x10 \x01(\x01\x12\x1b\n\x13\x61ltitude_coordinate\x18\x11 \x01(\x01\x12\x34\n\x10\x63urrent_time_utc\x18\x12 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x12\x43\n\x13\x64\x65vice_status_flags\x18\x17 \x03(\x0e\x32&.blueye.protocol.AquaTrollDeviceStatus\x12\x1a\n\x12used_battery_ticks\x18\x14 \x01(\r\x12\x1c\n\x14used_data_log_memory\x18\x15 \x01(\r\x12\x31\n\x07sensors\x18\x16 \x03(\x0e\x32 .blueye.protocol.AquaTrollSensorJ\x04\x08\x13\x10\x14R\rdevice_status\"\x94\x01\n\x19\x41quaTrollSensorParameters\x12\x33\n\tsensor_id\x18\x02 \x01(\x0e\x32 .blueye.protocol.AquaTrollSensor\x12\x42\n\x10parameter_blocks\x18\x03 \x03(\x0b\x32(.blueye.protocol.AquaTrollParameterBlock\"\x8c\x01\n\x1e\x41quaTrollSensorParametersArray\x12-\n\ttimestamp\x18\x01 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x12;\n\x07sensors\x18\x02 \x03(\x0b\x32*.blueye.protocol.AquaTrollSensorParameters\"\xbc\x01\n\x19SetAquaTrollParameterUnit\x12\x33\n\tsensor_id\x18\x01 \x01(\x0e\x32 .blueye.protocol.AquaTrollSensor\x12\x39\n\x0cparameter_id\x18\x02 \x01(\x0e\x32#.blueye.protocol.AquaTrollParameter\x12/\n\x07unit_id\x18\x03 \x01(\x0e\x32\x1e.blueye.protocol.AquaTrollUnit\"1\n\x1cSetAquaTrollConnectionStatus\x12\x11\n\tconnected\x18\x01 \x01(\x08*\xc1\x01\n\x04Type\x12\x14\n\x10TYPE_UNSPECIFIED\x10\x00\x12\x0e\n\nTYPE_SHORT\x10\x01\x12\x17\n\x13TYPE_UNSIGNED_SHORT\x10\x02\x12\r\n\tTYPE_LONG\x10\x03\x12\x16\n\x12TYPE_UNSIGNED_LONG\x10\x04\x12\x0e\n\nTYPE_FLOAT\x10\x05\x12\x0f\n\x0bTYPE_DOUBLE\x10\x06\x12\x12\n\x0eTYPE_CHARACTER\x10\x07\x12\x0f\n\x0bTYPE_STRING\x10\x08\x12\r\n\tTYPE_TIME\x10\t*\x85\x06\n\x0f\x41quaTrollDevice\x12!\n\x1d\x41QUA_TROLL_DEVICE_UNSPECIFIED\x10\x00\x12%\n!AQUA_TROLL_DEVICE_LEVEL_TROLL_500\x10\x01\x12%\n!AQUA_TROLL_DEVICE_LEVEL_TROLL_700\x10\x02\x12#\n\x1f\x41QUA_TROLL_DEVICE_BAROTROLL_500\x10\x03\x12%\n!AQUA_TROLL_DEVICE_LEVEL_TROLL_300\x10\x04\x12$\n AQUA_TROLL_DEVICE_AQUA_TROLL_200\x10\x05\x12$\n AQUA_TROLL_DEVICE_AQUA_TROLL_600\x10\x07\x12$\n AQUA_TROLL_DEVICE_AQUA_TROLL_100\x10\n\x12$\n AQUA_TROLL_DEVICE_FLOW_TROLL_500\x10\x0b\x12\x1d\n\x19\x41QUA_TROLL_DEVICE_RDO_PRO\x10\x0c\x12&\n\"AQUA_TROLL_DEVICE_RUGGED_TROLL_200\x10\x10\x12&\n\"AQUA_TROLL_DEVICE_RUGGED_BAROTROLL\x10\x11\x12$\n AQUA_TROLL_DEVICE_AQUA_TROLL_400\x10\x12\x12\x1f\n\x1b\x41QUA_TROLL_DEVICE_RDO_TITAN\x10\x13\x12\x1f\n\x1b\x41QUA_TROLL_DEVICE_SMARTROLL\x10\x15\x12+\n\'AQUA_TROLL_DEVICE_AQUA_TROLL_600_VENTED\x10\x1a\x12%\n!AQUA_TROLL_DEVICE_LEVEL_TROLL_400\x10\x1e\x12\x1f\n\x1b\x41QUA_TROLL_DEVICE_RDO_PRO_X\x10\x1f\x12$\n AQUA_TROLL_DEVICE_AQUA_TROLL_500\x10!\x12+\n\'AQUA_TROLL_DEVICE_AQUA_TROLL_500_VENTED\x10\"*\xb0\x02\n\x10\x41quaTrollQuality\x12\x1d\n\x19\x41QUA_TROLL_QUALITY_NORMAL\x10\x00\x12\'\n#AQUA_TROLL_QUALITY_USER_CAL_EXPIRED\x10\x01\x12*\n&AQUA_TROLL_QUALITY_FACTORY_CAL_EXPIRED\x10\x02\x12\x1c\n\x18\x41QUA_TROLL_QUALITY_ERROR\x10\x03\x12\x1e\n\x1a\x41QUA_TROLL_QUALITY_WARM_UP\x10\x04\x12%\n!AQUA_TROLL_QUALITY_SENSOR_WARNING\x10\x05\x12\"\n\x1e\x41QUA_TROLL_QUALITY_CALIBRATING\x10\x06\x12\x1f\n\x1b\x41QUA_TROLL_QUALITY_OFF_LINE\x10\x07*\xa3\x16\n\x12\x41quaTrollParameter\x12$\n AQUA_TROLL_PARAMETER_UNSPECIFIED\x10\x00\x12$\n AQUA_TROLL_PARAMETER_TEMPERATURE\x10\x01\x12!\n\x1d\x41QUA_TROLL_PARAMETER_PRESSURE\x10\x02\x12\x1e\n\x1a\x41QUA_TROLL_PARAMETER_DEPTH\x10\x03\x12-\n)AQUA_TROLL_PARAMETER_LEVEL_DEPTH_TO_WATER\x10\x04\x12\x30\n,AQUA_TROLL_PARAMETER_LEVEL_SURFACE_ELEVATION\x10\x05\x12!\n\x1d\x41QUA_TROLL_PARAMETER_LATITUDE\x10\x06\x12\"\n\x1e\x41QUA_TROLL_PARAMETER_LONGITUDE\x10\x07\x12\"\n\x1e\x41QUA_TROLL_PARAMETER_ELEVATION\x10\x08\x12,\n(AQUA_TROLL_PARAMETER_ACTUAL_CONDUCTIVITY\x10\t\x12.\n*AQUA_TROLL_PARAMETER_SPECIFIC_CONDUCTIVITY\x10\n\x12$\n AQUA_TROLL_PARAMETER_RESISTIVITY\x10\x0b\x12!\n\x1d\x41QUA_TROLL_PARAMETER_SALINITY\x10\x0c\x12/\n+AQUA_TROLL_PARAMETER_TOTAL_DISSOLVED_SOLIDS\x10\r\x12)\n%AQUA_TROLL_PARAMETER_DENSITY_OF_WATER\x10\x0e\x12)\n%AQUA_TROLL_PARAMETER_SPECIFIC_GRAVITY\x10\x0f\x12,\n(AQUA_TROLL_PARAMETER_BAROMETRIC_PRESSURE\x10\x10\x12\x1b\n\x17\x41QUA_TROLL_PARAMETER_PH\x10\x11\x12\x1e\n\x1a\x41QUA_TROLL_PARAMETER_PH_MV\x10\x12\x12\x1c\n\x18\x41QUA_TROLL_PARAMETER_ORP\x10\x13\x12\x37\n3AQUA_TROLL_PARAMETER_DISSOLVED_OXYGEN_CONCENTRATION\x10\x14\x12\x34\n0AQUA_TROLL_PARAMETER_DISSOLVED_OXYGEN_SATURATION\x10\x15\x12 \n\x1c\x41QUA_TROLL_PARAMETER_NITRATE\x10\x16\x12!\n\x1d\x41QUA_TROLL_PARAMETER_AMMONIUM\x10\x17\x12!\n\x1d\x41QUA_TROLL_PARAMETER_CHLORIDE\x10\x18\x12\"\n\x1e\x41QUA_TROLL_PARAMETER_TURBIDITY\x10\x19\x12(\n$AQUA_TROLL_PARAMETER_BATTERY_VOLTAGE\x10\x1a\x12\x1d\n\x19\x41QUA_TROLL_PARAMETER_HEAD\x10\x1b\x12\x1d\n\x19\x41QUA_TROLL_PARAMETER_FLOW\x10\x1c\x12#\n\x1f\x41QUA_TROLL_PARAMETER_TOTAL_FLOW\x10\x1d\x12\x30\n,AQUA_TROLL_PARAMETER_OXYGEN_PARTIAL_PRESSURE\x10\x1e\x12/\n+AQUA_TROLL_PARAMETER_TOTAL_SUSPENDED_SOLIDS\x10\x1f\x12)\n%AQUA_TROLL_PARAMETER_EXTERNAL_VOLTAGE\x10 \x12\x33\n/AQUA_TROLL_PARAMETER_BATTERY_CAPACITY_REMAINING\x10!\x12\x33\n/AQUA_TROLL_PARAMETER_RHODAMINE_WT_CONCENTRATION\x10\"\x12<\n8AQUA_TROLL_PARAMETER_RHODAMINE_WT_FLUORESCENCE_INTENSITY\x10#\x12\'\n#AQUA_TROLL_PARAMETER_CHLORIDE_CL_MV\x10$\x12@\n<AQUA_TROLL_PARAMETER_NITRATE_AS_NITROGEN_NO3_N_CONCENTRATION\x10%\x12\'\n#AQUA_TROLL_PARAMETER_NITRATE_NO3_MV\x10&\x12\x46\nBAQUA_TROLL_PARAMETER_AMMONIUM_AS_NITROGEN_NH4_PLUS_N_CONCENTRATION\x10\'\x12(\n$AQUA_TROLL_PARAMETER_AMMONIUM_NH4_MV\x10(\x12@\n<AQUA_TROLL_PARAMETER_AMMONIA_AS_NITROGEN_NH3_N_CONCENTRATION\x10)\x12\x46\nBAQUA_TROLL_PARAMETER_TOTAL_AMMONIA_AS_NITROGEN_NH3_N_CONCENTRATION\x10*\x12\x1b\n\x17\x41QUA_TROLL_PARAMETER_EH\x10\x30\x12!\n\x1d\x41QUA_TROLL_PARAMETER_VELOCITY\x10\x31\x12\x34\n0AQUA_TROLL_PARAMETER_CHLOROPHYLL_A_CONCENTRATION\x10\x32\x12=\n9AQUA_TROLL_PARAMETER_CHLOROPHYLL_A_FLUORESCENCE_INTENSITY\x10\x33\x12\x43\n?AQUA_TROLL_PARAMETER_BLUE_GREEN_ALGAE_PHYCOCYANIN_CONCENTRATION\x10\x36\x12L\nHAQUA_TROLL_PARAMETER_BLUE_GREEN_ALGAE_PHYCOCYANIN_FLUORESCENCE_INTENSITY\x10\x37\x12\x45\nAAQUA_TROLL_PARAMETER_BLUE_GREEN_ALGAE_PHYCOERYTHRIN_CONCENTRATION\x10:\x12N\nJAQUA_TROLL_PARAMETER_BLUE_GREEN_ALGAE_PHYCOERYTHRIN_FLUORESCENCE_INTENSITY\x10;\x12\x35\n1AQUA_TROLL_PARAMETER_FLUORESCEIN_WT_CONCENTRATION\x10\x43\x12>\n:AQUA_TROLL_PARAMETER_FLUORESCEIN_WT_FLUORESCENCE_INTENSITY\x10\x44\x12K\nGAQUA_TROLL_PARAMETER_FLUORESCENT_DISSOLVED_ORGANIC_MATTER_CONCENTRATION\x10\x45\x12T\nPAQUA_TROLL_PARAMETER_FLUORESCENT_DISSOLVED_ORGANIC_MATTER_FLUORESCENCE_INTENSITY\x10\x46\x12\x30\n,AQUA_TROLL_PARAMETER_CRUDE_OIL_CONCENTRATION\x10P\x12\x39\n5AQUA_TROLL_PARAMETER_CRUDE_OIL_FLUORESCENCE_INTENSITY\x10Q\x12G\nCAQUA_TROLL_PARAMETER_COLORED_DISSOLVED_ORGANIC_MATTER_CONCENTRATION\x10W*\xbc\x19\n\rAquaTrollUnit\x12\x1f\n\x1b\x41QUA_TROLL_UNIT_UNSPECIFIED\x10\x00\x12 \n\x1c\x41QUA_TROLL_UNIT_TEMP_CELSIUS\x10\x01\x12\"\n\x1e\x41QUA_TROLL_UNIT_TEMP_FARENHEIT\x10\x02\x12\x1f\n\x1b\x41QUA_TROLL_UNIT_TEMP_KELVIN\x10\x03\x12*\n&AQUA_TROLL_UNIT_POUNDS_PER_SQUARE_INCH\x10\x11\x12\x1b\n\x17\x41QUA_TROLL_UNIT_PASCALS\x10\x12\x12\x1f\n\x1b\x41QUA_TROLL_UNIT_KILOPASCALS\x10\x13\x12\x18\n\x14\x41QUA_TROLL_UNIT_BARS\x10\x14\x12\x1d\n\x19\x41QUA_TROLL_UNIT_MILLIBARS\x10\x15\x12*\n&AQUA_TROLL_UNIT_MILLIMETERS_OF_MERCURY\x10\x16\x12%\n!AQUA_TROLL_UNIT_INCHES_OF_MERCURY\x10\x17\x12(\n$AQUA_TROLL_UNIT_CENTIMETERS_OF_WATER\x10\x18\x12#\n\x1f\x41QUA_TROLL_UNIT_INCHES_OF_WATER\x10\x19\x12\x18\n\x14\x41QUA_TROLL_UNIT_TORR\x10\x1a\x12\'\n#AQUA_TROLL_UNIT_STANDARD_ATMOSPHERE\x10\x1b\x12\x1f\n\x1b\x41QUA_TROLL_UNIT_MILLIMETERS\x10!\x12\x1f\n\x1b\x41QUA_TROLL_UNIT_CENTIMETERS\x10\"\x12\x1a\n\x16\x41QUA_TROLL_UNIT_METERS\x10#\x12\x1d\n\x19\x41QUA_TROLL_UNIT_KILOMETER\x10$\x12\x1a\n\x16\x41QUA_TROLL_UNIT_INCHES\x10%\x12\x18\n\x14\x41QUA_TROLL_UNIT_FEET\x10&\x12\x1b\n\x17\x41QUA_TROLL_UNIT_DEGREES\x10\x31\x12\x1b\n\x17\x41QUA_TROLL_UNIT_MINUTES\x10\x32\x12\x1b\n\x17\x41QUA_TROLL_UNIT_SECONDS\x10\x33\x12/\n+AQUA_TROLL_UNIT_MICROSIEMENS_PER_CENTIMETER\x10\x41\x12/\n+AQUA_TROLL_UNIT_MILLISIEMENS_PER_CENTIMETER\x10\x42\x12#\n\x1f\x41QUA_TROLL_UNIT_OHM_CENTIMETERS\x10Q\x12,\n(AQUA_TROLL_UNIT_PRACTICAL_SALINITY_UNITS\x10\x61\x12/\n+AQUA_TROLL_UNIT_PARTS_PER_THOUSAND_SALINITY\x10\x62\x12%\n!AQUA_TROLL_UNIT_PARTS_PER_MILLION\x10q\x12&\n\"AQUA_TROLL_UNIT_PARTS_PER_THOUSAND\x10r\x12.\n*AQUA_TROLL_UNIT_PARTS_PER_MILLION_NITROGEN\x10s\x12.\n*AQUA_TROLL_UNIT_PARTS_PER_MILLION_CHLORIDE\x10t\x12(\n$AQUA_TROLL_UNIT_MILLIGRAMS_PER_LITER\x10u\x12(\n$AQUA_TROLL_UNIT_MICROGRAMS_PER_LITER\x10v\x12\x33\n/AQUA_TROLL_UNIT_MICROMOLES_PER_LITER_DEPRECATED\x10w\x12#\n\x1f\x41QUA_TROLL_UNIT_GRAMS_PER_LITER\x10x\x12%\n!AQUA_TROLL_UNIT_PARTS_PER_BILLION\x10y\x12/\n*AQUA_TROLL_UNIT_GRAMS_PER_CUBIC_CENTIMETER\x10\x81\x01\x12\x17\n\x12\x41QUA_TROLL_UNIT_PH\x10\x91\x01\x12 \n\x1b\x41QUA_TROLL_UNIT_MICRO_VOLTS\x10\xa1\x01\x12 \n\x1b\x41QUA_TROLL_UNIT_MILLI_VOLTS\x10\xa2\x01\x12\x1a\n\x15\x41QUA_TROLL_UNIT_VOLTS\x10\xa3\x01\x12\'\n\"AQUA_TROLL_UNIT_PERCENT_SATURATION\x10\xb1\x01\x12\x31\n,AQUA_TROLL_UNIT_FORMAZIN_NEPHELOMETRIC_UNITS\x10\xc1\x01\x12\x32\n-AQUA_TROLL_UNIT_NEPHELOMETRIC_TURBIDITY_UNITS\x10\xc2\x01\x12-\n(AQUA_TROLL_UNIT_FORMAZIN_TURBIDITY_UNITS\x10\xc3\x01\x12*\n%AQUA_TROLL_UNIT_CUBIC_FEET_PER_SECOND\x10\xd1\x01\x12*\n%AQUA_TROLL_UNIT_CUBIC_FEET_PER_MINUTE\x10\xd2\x01\x12(\n#AQUA_TROLL_UNIT_CUBIC_FEET_PER_HOUR\x10\xd3\x01\x12\'\n\"AQUA_TROLL_UNIT_CUBIC_FEET_PER_DAY\x10\xd4\x01\x12\'\n\"AQUA_TROLL_UNIT_GALLONS_PER_SECOND\x10\xd5\x01\x12\'\n\"AQUA_TROLL_UNIT_GALLONS_PER_MINUTE\x10\xd6\x01\x12%\n AQUA_TROLL_UNIT_GALLONS_PER_HOUR\x10\xd7\x01\x12\x30\n+AQUA_TROLL_UNIT_MILLIONS_OF_GALLONS_PER_DAY\x10\xd8\x01\x12,\n\'AQUA_TROLL_UNIT_CUBIC_METERS_PER_SECOND\x10\xd9\x01\x12,\n\'AQUA_TROLL_UNIT_CUBIC_METERS_PER_MINUTE\x10\xda\x01\x12*\n%AQUA_TROLL_UNIT_CUBIC_METERS_PER_HOUR\x10\xdb\x01\x12)\n$AQUA_TROLL_UNIT_CUBIC_METERS_PER_DAY\x10\xdc\x01\x12&\n!AQUA_TROLL_UNIT_LITERS_PER_SECOND\x10\xdd\x01\x12/\n*AQUA_TROLL_UNIT_MILLIONS_OF_LITERS_PER_DAY\x10\xde\x01\x12+\n&AQUA_TROLL_UNIT_MILLILITERS_PER_MINUTE\x10\xdf\x01\x12\x30\n+AQUA_TROLL_UNIT_THOUSANDS_OF_LITERS_PER_DAY\x10\xe0\x01\x12\x1f\n\x1a\x41QUA_TROLL_UNIT_CUBIC_FEET\x10\xe1\x01\x12\x1c\n\x17\x41QUA_TROLL_UNIT_GALLONS\x10\xe2\x01\x12(\n#AQUA_TROLL_UNIT_MILLIONS_OF_GALLONS\x10\xe3\x01\x12!\n\x1c\x41QUA_TROLL_UNIT_CUBIC_METERS\x10\xe4\x01\x12\x1b\n\x16\x41QUA_TROLL_UNIT_LITERS\x10\xe5\x01\x12\x1e\n\x19\x41QUA_TROLL_UNIT_ACRE_FEET\x10\xe6\x01\x12 \n\x1b\x41QUA_TROLL_UNIT_MILLILITERS\x10\xe7\x01\x12\'\n\"AQUA_TROLL_UNIT_MILLIONS_OF_LITERS\x10\xe8\x01\x12(\n#AQUA_TROLL_UNIT_THOUSANDS_OF_LITERS\x10\xe9\x01\x12 \n\x1b\x41QUA_TROLL_UNIT_ACRE_INCHES\x10\xea\x01\x12\x1c\n\x17\x41QUA_TROLL_UNIT_PERCENT\x10\xf1\x01\x12\x30\n+AQUA_TROLL_UNIT_RELATIVE_FLUORESCENCE_UNITS\x10\x81\x02\x12+\n&AQUA_TROLL_UNIT_MILLILITERS_PER_SECOND\x10\x91\x02\x12)\n$AQUA_TROLL_UNIT_MILLILITERS_PER_HOUR\x10\x92\x02\x12&\n!AQUA_TROLL_UNIT_LITERS_PER_MINUTE\x10\x93\x02\x12$\n\x1f\x41QUA_TROLL_UNIT_LITERS_PER_HOUR\x10\x94\x02\x12\x1e\n\x19\x41QUA_TROLL_UNIT_MICROAMPS\x10\xa1\x02\x12\x1e\n\x19\x41QUA_TROLL_UNIT_MILLIAMPS\x10\xa2\x02\x12\x19\n\x14\x41QUA_TROLL_UNIT_AMPS\x10\xa3\x02\x12$\n\x1f\x41QUA_TROLL_UNIT_FEET_PER_SECOND\x10\xb1\x02\x12&\n!AQUA_TROLL_UNIT_METERS_PER_SECOND\x10\xb2\x02*\xd2\x1f\n\x0f\x41quaTrollSensor\x12!\n\x1d\x41QUA_TROLL_SENSOR_UNSPECIFIED\x10\x00\x12!\n\x1d\x41QUA_TROLL_SENSOR_TEMPERATURE\x10\x01\x12Q\nMAQUA_TROLL_SENSOR_S5_PSI_FULL_SCALE_GAUGE_PRESSURE_WITH_LEVEL_AND_TEMPERATURE\x10\x02\x12R\nNAQUA_TROLL_SENSOR_S15_PSI_FULL_SCALE_GAUGE_PRESSURE_WITH_LEVEL_AND_TEMPERATURE\x10\x03\x12R\nNAQUA_TROLL_SENSOR_S30_PSI_FULL_SCALE_GAUGE_PRESSURE_WITH_LEVEL_AND_TEMPERATURE\x10\x04\x12S\nOAQUA_TROLL_SENSOR_S100_PSI_FULL_SCALE_GAUGE_PRESSURE_WITH_LEVEL_AND_TEMPERATURE\x10\x05\x12S\nOAQUA_TROLL_SENSOR_S300_PSI_FULL_SCALE_GAUGE_PRESSURE_WITH_LEVEL_AND_TEMPERATURE\x10\x06\x12S\nOAQUA_TROLL_SENSOR_S500_PSI_FULL_SCALE_GAUGE_PRESSURE_WITH_LEVEL_AND_TEMPERATURE\x10\x07\x12W\nSAQUA_TROLL_SENSOR_S1000_PSI_FULL_SCALE_ABSOLUTE_PRESSURE_WITH_LEVEL_AND_TEMPERATURE\x10\x08\x12U\nQAQUA_TROLL_SENSOR_S30_PSI_FULL_SCALE_ABSOLUTE_PRESSURE_WITH_LEVEL_AND_TEMPERATURE\x10\t\x12V\nRAQUA_TROLL_SENSOR_S100_PSI_FULL_SCALE_ABSOLUTE_PRESSURE_WITH_LEVEL_AND_TEMPERATURE\x10\n\x12V\nRAQUA_TROLL_SENSOR_S300_PSI_FULL_SCALE_ABSOLUTE_PRESSURE_WITH_LEVEL_AND_TEMPERATURE\x10\x0b\x12V\nRAQUA_TROLL_SENSOR_S500_PSI_FULL_SCALE_ABSOLUTE_PRESSURE_WITH_LEVEL_AND_TEMPERATURE\x10\x0c\x12K\nGAQUA_TROLL_SENSOR_S30_PSI_FULL_SCALE_ABSOLUTE_PRESSURE_WITH_TEMPERATURE\x10\r\x12^\nZAQUA_TROLL_SENSOR_S5_PSI_FULL_SCALE_GAUGE_PRESSURE_WITH_LEVEL_TEMPERATURE_AND_CONDUCTIVITY\x10\x0e\x12_\n[AQUA_TROLL_SENSOR_S15_PSI_FULL_SCALE_GAUGE_PRESSURE_WITH_LEVEL_TEMPERATURE_AND_CONDUCTIVITY\x10\x0f\x12_\n[AQUA_TROLL_SENSOR_S30_PSI_FULL_SCALE_GAUGE_PRESSURE_WITH_LEVEL_TEMPERATURE_AND_CONDUCTIVITY\x10\x10\x12`\n\\AQUA_TROLL_SENSOR_S100_PSI_FULL_SCALE_GAUGE_PRESSURE_WITH_LEVEL_TEMPERATURE_AND_CONDUCTIVITY\x10\x11\x12`\n\\AQUA_TROLL_SENSOR_S300_PSI_FULL_SCALE_GAUGE_PRESSURE_WITH_LEVEL_TEMPERATURE_AND_CONDUCTIVITY\x10\x12\x12`\n\\AQUA_TROLL_SENSOR_S500_PSI_FULL_SCALE_GAUGE_PRESSURE_WITH_LEVEL_TEMPERATURE_AND_CONDUCTIVITY\x10\x13\x12\x1e\n\x1a\x41QUA_TROLL_SENSOR_NOT_USED\x10\x14\x12\x62\n^AQUA_TROLL_SENSOR_S30_PSI_FULL_SCALE_ABSOLUTE_PRESSURE_WITH_LEVEL_TEMPERATURE_AND_CONDUCTIVITY\x10\x15\x12\x63\n_AQUA_TROLL_SENSOR_S100_PSI_FULL_SCALE_ABSOLUTE_PRESSURE_WITH_LEVEL_TEMPERATURE_AND_CONDUCTIVITY\x10\x16\x12\x63\n_AQUA_TROLL_SENSOR_S300_PSI_FULL_SCALE_ABSOLUTE_PRESSURE_WITH_LEVEL_TEMPERATURE_AND_CONDUCTIVITY\x10\x17\x12\x63\n_AQUA_TROLL_SENSOR_S500_PSI_FULL_SCALE_ABSOLUTE_PRESSURE_WITH_LEVEL_TEMPERATURE_AND_CONDUCTIVITY\x10\x18\x12;\n7AQUA_TROLL_SENSOR_S165_PSI_FULL_SCALE_ABSOLUTE_PRESSURE\x10\x19\x12&\n\"AQUA_TROLL_SENSOR_PH_ANALOG_SENSOR\x10\x1a\x12*\n&AQUA_TROLL_SENSOR_PH_ORP_ANALOG_SENSOR\x10\x1b\x12?\n;AQUA_TROLL_SENSOR_DISSOLVED_OXYGEN_CLARK_CELL_ANALOG_SENSOR\x10\x1c\x12+\n\'AQUA_TROLL_SENSOR_NITRATE_ANALOG_SENSOR\x10\x1d\x12,\n(AQUA_TROLL_SENSOR_AMMONIUM_ANALOG_SENSOR\x10\x1e\x12,\n(AQUA_TROLL_SENSOR_CHLORIDE_ANALOG_SENSOR\x10\x1f\x12W\nSAQUA_TROLL_SENSOR_S100_FOOT_FULL_SCALE_LEVEL_WITH_ABSOLUTE_PRESSURE_AND_TEMPERATURE\x10 \x12W\nSAQUA_TROLL_SENSOR_S250_FOOT_FULL_SCALE_LEVEL_WITH_ABSOLUTE_PRESSURE_AND_TEMPERATURE\x10!\x12V\nRAQUA_TROLL_SENSOR_S30_FOOT_FULL_SCALE_LEVEL_WITH_ABSOLUTE_PRESSURE_AND_TEMPERATURE\x10\"\x12\x32\n.AQUA_TROLL_SENSOR_CONDUCTIVITY_AND_TEMPERATURE\x10#\x12U\nQAQUA_TROLL_SENSOR_S5_PSI_FULL_SCALE_GAUGE_PRESSURE_WITH_TEMPERATURE_HEAD_AND_FLOW\x10$\x12V\nRAQUA_TROLL_SENSOR_S15_PSI_FULL_SCALE_GAUGE_PRESSURE_WITH_TEMPERATURE_HEAD_AND_FLOW\x10%\x12V\nRAQUA_TROLL_SENSOR_S30_PSI_FULL_SCALE_GAUGE_PRESSURE_WITH_TEMPERATURE_HEAD_AND_FLOW\x10&\x12W\nSAQUA_TROLL_SENSOR_S100_PSI_FULL_SCALE_GAUGE_PRESSURE_WITH_TEMPERATURE_HEAD_AND_FLOW\x10\'\x12W\nSAQUA_TROLL_SENSOR_S300_PSI_FULL_SCALE_GAUGE_PRESSURE_WITH_TEMPERATURE_HEAD_AND_FLOW\x10(\x12W\nSAQUA_TROLL_SENSOR_S500_PSI_FULL_SCALE_GAUGE_PRESSURE_WITH_TEMPERATURE_HEAD_AND_FLOW\x10)\x12?\n;AQUA_TROLL_SENSOR_OPTICAL_DISSOLVED_OXYGEN_WITH_TEMPERATURE\x10*\x12\x1c\n\x18\x41QUA_TROLL_SENSOR_S1_BAR\x10+\x12\x1c\n\x18\x41QUA_TROLL_SENSOR_S2_BAR\x10,\x12\x1c\n\x18\x41QUA_TROLL_SENSOR_S5_BAR\x10-\x12&\n\"AQUA_TROLL_SENSOR_TURBIDITY_SENSOR\x10\x32\x12(\n$AQUA_TROLL_SENSOR_TEMPERATURE_SENSOR\x10\x37\x12)\n%AQUA_TROLL_SENSOR_CONDUCTIVITY_SENSOR\x10\x38\x12 \n\x1c\x41QUA_TROLL_SENSOR_RDO_SENSOR\x10\x39\x12#\n\x1f\x41QUA_TROLL_SENSOR_PH_ORP_SENSOR\x10:\x12)\n%AQUA_TROLL_SENSOR_RHODAMINE_WT_SENSOR\x10<\x12*\n&AQUA_TROLL_SENSOR_CHLOROPHYLL_A_SENSOR\x10>\x12\x39\n5AQUA_TROLL_SENSOR_BLUE_GREEN_ALGAE_PHYCOCYANIN_SENSOR\x10@\x12;\n7AQUA_TROLL_SENSOR_BLUE_GREEN_ALGAE_PHYCOERYTHRIN_SENSOR\x10\x41\x12(\n$AQUA_TROLL_SENSOR_NITRATE_ISE_SENSOR\x10\x46\x12)\n%AQUA_TROLL_SENSOR_AMMONIUM_ISE_SENSOR\x10G\x12)\n%AQUA_TROLL_SENSOR_CHLORIDE_ISE_SENSOR\x10H\x12&\n\"AQUA_TROLL_SENSOR_PROBE_PARAMETERS\x10O*\xa3\x03\n\x15\x41quaTrollSensorStatus\x12.\n*AQUA_TROLL_SENSOR_STATUS_SENSOR_HIGH_ALARM\x10\x00\x12\x30\n,AQUA_TROLL_SENSOR_STATUS_SENSOR_HIGH_WARNING\x10\x01\x12/\n+AQUA_TROLL_SENSOR_STATUS_SENSOR_LOW_WARNING\x10\x02\x12-\n)AQUA_TROLL_SENSOR_STATUS_SENSOR_LOW_ALARM\x10\x03\x12\x37\n3AQUA_TROLL_SENSOR_STATUS_SENSOR_CALIBRATION_WARNING\x10\x04\x12/\n+AQUA_TROLL_SENSOR_STATUS_SENSOR_MALFUNCTION\x10\x05\x12.\n*AQUA_TROLL_SENSOR_STATUS_SENSOR_MODE_BIT_1\x10\x08\x12.\n*AQUA_TROLL_SENSOR_STATUS_SENSOR_MODE_BIT_2\x10\t*\x9a\x05\n\x15\x41quaTrollDeviceStatus\x12.\n*AQUA_TROLL_DEVICE_STATUS_SENSOR_HIGH_ALARM\x10\x00\x12\x30\n,AQUA_TROLL_DEVICE_STATUS_SENSOR_HIGH_WARNING\x10\x01\x12/\n+AQUA_TROLL_DEVICE_STATUS_SENSOR_LOW_WARNING\x10\x02\x12-\n)AQUA_TROLL_DEVICE_STATUS_SENSOR_LOW_ALARM\x10\x03\x12\x37\n3AQUA_TROLL_DEVICE_STATUS_SENSOR_CALIBRATION_WARNING\x10\x04\x12/\n+AQUA_TROLL_DEVICE_STATUS_SENSOR_MALFUNCTION\x10\x05\x12\x36\n2AQUA_TROLL_DEVICE_STATUS_POWER_MANAGEMENT_DISABLED\x10\x08\x12,\n(AQUA_TROLL_DEVICE_STATUS_DEVICE_OFF_LINE\x10\t\x12;\n7AQUA_TROLL_DEVICE_STATUS_DEVICE_HARDWARE_RESET_OCCURRED\x10\n\x12/\n+AQUA_TROLL_DEVICE_STATUS_DEVICE_MALFUNCTION\x10\x0b\x12.\n*AQUA_TROLL_DEVICE_STATUS_NO_EXTERNAL_POWER\x10\x0c\x12(\n$AQUA_TROLL_DEVICE_STATUS_LOW_BATTERY\x10\r\x12\'\n#AQUA_TROLL_DEVICE_STATUS_LOW_MEMORY\x10\x0e\x42\x1b\xaa\x02\x18\x42lueye.Protocol.Protobufb\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'aquatroll_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'\252\002\030Blueye.Protocol.Protobuf'
  _globals['_TYPE']._serialized_start=2956
  _globals['_TYPE']._serialized_end=3149
  _globals['_AQUATROLLDEVICE']._serialized_start=3152
  _globals['_AQUATROLLDEVICE']._serialized_end=3925
  _globals['_AQUATROLLQUALITY']._serialized_start=3928
  _globals['_AQUATROLLQUALITY']._serialized_end=4232
  _globals['_AQUATROLLPARAMETER']._serialized_start=4235
  _globals['_AQUATROLLPARAMETER']._serialized_end=7086
  _globals['_AQUATROLLUNIT']._serialized_start=7089
  _globals['_AQUATROLLUNIT']._serialized_end=10349
  _globals['_AQUATROLLSENSOR']._serialized_start=10352
  _globals['_AQUATROLLSENSOR']._serialized_end=14402
  _globals['_AQUATROLLSENSORSTATUS']._serialized_start=14405
  _globals['_AQUATROLLSENSORSTATUS']._serialized_end=14824
  _globals['_AQUATROLLDEVICESTATUS']._serialized_start=14827
  _globals['_AQUATROLLDEVICESTATUS']._serialized_end=15493
  _globals['_AQUATROLLPARAMETERBLOCK']._serialized_start=70
  _globals['_AQUATROLLPARAMETERBLOCK']._serialized_end=402
  _globals['_AQUATROLLSENSORMETADATA']._serialized_start=405
  _globals['_AQUATROLLSENSORMETADATA']._serialized_end=1412
  _globals['_AQUATROLLSENSORMETADATAARRAY']._serialized_start=1415
  _globals['_AQUATROLLSENSORMETADATAARRAY']._serialized_end=1551
  _globals['_AQUATROLLPROBEMETADATA']._serialized_start=1554
  _globals['_AQUATROLLPROBEMETADATA']._serialized_end=2417
  _globals['_AQUATROLLSENSORPARAMETERS']._serialized_start=2420
  _globals['_AQUATROLLSENSORPARAMETERS']._serialized_end=2568
  _globals['_AQUATROLLSENSORPARAMETERSARRAY']._serialized_start=2571
  _globals['_AQUATROLLSENSORPARAMETERSARRAY']._serialized_end=2711
  _globals['_SETAQUATROLLPARAMETERUNIT']._serialized_start=2714
  _globals['_SETAQUATROLLPARAMETERUNIT']._serialized_end=2902
  _globals['_SETAQUATROLLCONNECTIONSTATUS']._serialized_start=2904
  _globals['_SETAQUATROLLCONNECTIONSTATUS']._serialized_end=2953
# @@protoc_insertion_point(module_scope)