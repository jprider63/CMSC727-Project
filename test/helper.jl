# Imports a specified time series dataset
# Other than the household power dataset, all are in same format
function importData(datasetName)
	filesep = "/"
	if OS_NAME == ":Windows"
		filesep = "\\"
	end
	testPath = pwd() * filesep * ".." * filesep * "test" * filesep

	if datasetName == "household"
		csvPath = testPath * "household_power_consumption.txt"
		data = readdlm(csvPath, ';', Any)
		# Strips non float values (1st and 2nd column)
		data = data[:, 3:]
	else
		csvPath = testPath * datasetName * ".csv"
		data = readcsv(csvPath, Float64)
	end

	data = data[2:, :]	# Strips column names (1st row)
	return data
end

