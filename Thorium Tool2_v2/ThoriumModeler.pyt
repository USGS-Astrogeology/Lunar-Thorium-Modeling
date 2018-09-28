import numpy as np
import math
import scipy.integrate
from scipy.interpolate import interp1d

#Debug
import arcpy

def deleteinmem(name):
    """
    Delete an 'in_memory/<name>' raster layer
    if it exists.
    """
    try:
        arcpy.Delete_management(name)
    except:
        pass
    return

def computeedgecoords(rasterinfo):
    """
    Compute the lat and long coordinates bounding the area
    """

    height = rasterinfo['height']
    width = rasterinfo['width']
    xmin = rasterinfo['xmin']
    ymin = rasterinfo['ymin']
    cellheight = rasterinfo['cellheight']
    cellwidth = rasterinfo['cellwidth']

    xcoords = []
    for x in range(width):
        xcoords.append(xmin + (cellwidth * x) + (cellwidth / 2))
    ycoords = []
    for y in range(height):
        ycoords.append(ymin + (cellheight * y) + (cellwidth / 2))

    return np.array(xcoords), np.array(ycoords)

def idl_tabulate(x, f, p=5) :
    """
    Perform a p point Newton-Cotes integration similar to the IDL function
    int_tabulated.

    Using the example from the IDL docstring the return value is 1.6271 with
    and actual value of 1.6405.  This function returns 1.6419.

    From http://stackoverflow.com/questions/14345001/idls-int-tabulate-scipy-equivalent
    """
    def newton_cotes(x, f) :
        if x.shape[0] < 2 :
            return 0
        rn = (x.shape[0] - 1) * (x - x[0]) / (x[-1] - x[0])
        weights = scipy.integrate.newton_cotes(rn)[0]
        return (x[-1] - x[0]) / (x.shape[0] - 1) * np.dot(weights, f)
    ret = 0
    for idx in xrange(0, x.shape[0], p - 1) :
        ret += newton_cotes(x[idx:idx + p], f[idx:idx + p])
    return ret

def computekappanormalization(sigma, kappa):
    """
    Compute the normalization of the kappa function.
    """
    skip = math.floor(sigma / 5.0)
    limit = math.floor(40.0 * sigma)
    num = math.floor(float(limit) / float(skip))
    xint = np.arange(num, dtype=np.float)
    xint = xint * skip - limit / 2.0
    yint = 1.0 * (1 + 0.5 * xint ** 2 / sigma ** 2) ** (-kappa - 1)
    normalization = idl_tabulate(xint,yint)
    return normalization

def table_distance_regional(llat, llon, radius):
    """
    At a fixed longitude, compute the distance between clon and
    all other lon for each latitude in the input, where

    clon = the longitude at center of the input raster.

    Yields a table of
    """
    arcpy.AddMessage('Generating distance table.')
    nlat = len(llat)
    nlon = len(llon)
    table_size = nlat * nlon

    #tableidx = np.empty((nlat, table_size), dtype = np.int32)
    #tableidx[:] = -1
    tablelon = np.empty((nlat, table_size), dtype = np.int32)
    tablelat = np.empty((nlat, table_size), dtype = np.int32)
    tablelon[:] = -1
    tablelat[:] = -1
    tabledist = np.empty((nlat, table_size), dtype = np.float32)
    tabledist[:] = -1
    npoints = np.zeros(nlat)

    rlon = np.radians(llon)
    rlat = np.radians(llat)
    lon0 = rlon[int(nlon / 2)]
    maxq = 0

    #Precompute everything we can.
    arcpy.AddMessage('Precomputing vectors for distance table generation.')
    sinlat = np.sin(rlat)
    coslat = np.cos(rlat)
    a3 = np.cos(rlon - lon0)
    arcpy.AddMessage('Computing latitude dependent pixel distances.')
    for k, lat0 in enumerate(rlat):
        a1 = sinlat * math.sin(lat0)
        a2 = coslat * math.cos(lat0)
        a4 = np.ones(nlon, dtype=np.float)

        dp = np.outer(a1, a4) + np.outer(a2, a3)
        a = dp.clip(min=-1,max=1)
        d = 1738.0 * np.arccos(a)  # Moon
        q = np.where(d < radius)
        if len(q[0]) > maxq:
            maxq = len(q[0])

        i = np.arange(len(q[0]), dtype=np.float)
        tablelon[k,i[0]:i[-1] + 1] = q[1]
        tablelat[k,i[0]:i[-1] + 1] = q[0]
        tabledist[k,i[0]:i[-1] + 1] = d[q]
        npoints[k] = len(q[0])
    table = {#'idx':tableidx[:,0:maxq-1],
             'lon':tablelon[:,0:maxq],
             'lat':tablelat[:,0:maxq],
             'dist':tabledist[:,0:maxq],
             'npoints':npoints}
    arcpy.AddMessage('Table generated.')
    return table

def kappapreprocessing(arr, sigma, rasterinfo, kappa, fill, cellsize, finite=True):
    """
    Preprocessing to compute the distance table and normalization scalar.
    """

    arcpy.AddMessage('Initiating Kappa Smoothing')
    #Setup the output

    '''
    #Set the fill value in the input
    mask = np.where(arr <= -99999)
    if len(mask) > 0:
        arcpy.AddMessage('Masking No Data Pixels with supplied fill value.')
        arr[mask] = fill
    '''

    #Radius to which cells are included
    radius = sigma * 5.0

    #Kappa normalization - needs scipy.integrate working
    normalization = computekappanormalization(sigma, kappa)

    #Approximate the lat and lon centroids of the gird cells.
    llon,llat = computeedgecoords(rasterinfo)
    nlat = len(llat)
    nlon = len(llon)

    #Get the distance matrix
    table = table_distance_regional(llat, llon, radius)

    #Call the smoothing function to generate derived products
    map0 = kappasmoothing(arr, table, nlon, nlat, sigma, kappa, normalization, finite)
    finalmap = kappasmoothing(map0, table, nlon, nlat, sigma, kappa, normalization, finite)

    return map0, finalmap

def kappasmoothing(arr, table, nlon, nlat, sigma, kappa, normalization, finite):
    """
    The smoothing function.
    """
    
    gridout = np.zeros(arr.shape, dtype=np.float32)
    #Apply the kernel
    for j in range(nlat):
        if j % 10 == 0:
            arcpy.AddMessage('Processed {} / {}.'.format(j, nlat))
        npts = int(table['npoints'][j])
        whr_lat = table['lat'][j,0:npts]  # Latitude Indices
        whr_lon_tmp = table['lon'][j,0:npts]  # Longitude Indices
        whr_dist = table['dist'][j,0:npts]  # Distances
	
	#arcpy.AddMessage((whr_lon_tmp[:100]))

        for i in range(nlon):
            clonidx = i - int(nlon / 2.0)
            whr_lon = whr_lon_tmp + clonidx
	    
	    if len(whr_lon[whr_lon > nlon - 1]) > 0:
		whr_lon[whr_lon > nlon - 1] = whr_lon[whr_lon > nlon -1] - nlon

            #arcpy.AddMessage(whr_lat)
            #arcpy.AddMessage(whr_lon)
	    #arcpy.AddMessage(arr.shape)
            #arcpy.AddMessage(whr_dist)
            y = arr[whr_lat,whr_lon]
            if finite:
                q = np.where(np.isfinite(y) == True)[0]
            else:
                q = np.where(y > 0)[0]

            if len(q) > 0:
                #Why is this multiplied by 1.0?
                #f = 1.0 * (1.0 + 0.5 * whr_dist[q] ** 2 / sigma ** 2) ** (-kappa - 1)
                f =  1.0 * (1.0 + 0.5 * whr_dist[q] ** 2 / sigma ** 2) ** (-kappa - 1)
                f /= normalization
                gridout[j,i] = np.sum(y[q] * f) / np.sum(f)


    return gridout
    #Working to here.

class Toolbox(object):
    def __init__(self):
        """Define the toolbox (the name of the toolbox is the name of the
        .pyt file)."""
        self.label = "ThoriumModeler"
        self.alias = ""

        # List of tool classes associated with this toolbox
        self.tools = [ThoriumModelOnce, ChiSqDifference]


class ChiSqDifference(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Chi Squared Difference"
        self.description = ""
        self.canRunInBackground = False
	
    def getParameterInfo(self):
        """Define parameter definitions"""
	#Workspace
	workspace = arcpy.Parameter(
	    displayName="Workspace",
	    name="workspace",
	    datatype="DEWorkspace",
	    parameterType="Required",
	    direction="Input")
	workspace.defaultEnvironmentName = "workspace"
	
        # Thorium Raster
        thorium_raster = arcpy.Parameter(
            displayName="Thorium Base",
            name="thorium_raster",
            datatype="GPRasterLayer",
            parameterType="Required",
            direction="Input")
	
        # Uncertainty Raster
        uncertraster = arcpy.Parameter(
            displayName="Thorium Uncertainty",
            name="uncertraster",
            datatype="GPRasterLayer",
            parameterType="Optional",
            direction="Input")	
	
	inputs = arcpy.Parameter(
	    displayName='Input Features',
	    name='in_features',
	    datatype='GPValueTable',
	    parameterType='Required',
	    direction='Input')

	inputs.columns = [['GPRasterLayer', 'Final Map'], ['String', 'Name']]

        params = [workspace, thorium_raster, uncertraster, inputs]
        return params

    def isLicensed(self):
        """Set whether tool is licensed to execute."""
        return True

    def updateParameters(self, parameters):
        """Modify the values and properties of parameters before internal
        validation is performed.  This method is called whenever a parameter
        has been changed."""

        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter.  This method is called after internal validation."""
        return

    def execute(self, parameters, messages):
	workspace = parameters[0].valueAsText
        thmodel = parameters[1].valueAsText
        uncert = parameters[2].valueAsText
	input_difference_maps = parameters[3].values
	
	#Set the overwrite ENV variable to true
	arcpy.env.overwriteOutput = True
	#Set the workspace
	arcpy.env.workspace = workspace	
	
	#Get the map and dataframe 0
	mxd = arcpy.mapping.MapDocument("CURRENT")
	df = arcpy.mapping.ListDataFrames(mxd, "*")[0]	
	
	#Clip out the thorium base and uncertainty base
	thbase_raster = arcpy.Clip_management(thmodel,
	                                      '#',
	                                      'thbase',
	                                      input_difference_maps[0][0])
	thbase = arcpy.RasterToNumPyArray('thbase')
	
	if uncert != None:
	    uncertbase_raster = arcpy.Clip_management(uncert,
	                                              '#',
	                                              'uncertbase',
	                                              input_difference_maps[0][0])
	    uncertbase = arcpy.RasterToNumPyArray('uncertbase')
	

	xlabels = []
	chisq = []
	
	#Get the cell x,y coordinates using the first input
	rasterdesc = arcpy.Describe(str(input_difference_maps[0][0]))
	xmin = rasterdesc.Extent.XMin
	ymin = rasterdesc.Extent.YMin
	#Get cell information
	cellheight = rasterdesc.MeanCellHeight
	cellwidth = rasterdesc.MeanCellWidth
	height = rasterdesc.height
	width = rasterdesc.width
	lowerleft = arcpy.Point(xmin, ymin)	
	
	
	for idx, i in enumerate(input_difference_maps):
	    arcpy.AddMessage("Processing map {} / {}".format(idx + 1, len(input_difference_maps)))
	    finalmap = arcpy.RasterToNumPyArray(str(i[0]))
	    outputname = i[1]
	    
	    #Compute the global difference
	    #If uncertainty is provided compute the diff using that as well		
	    if uncert != None:
		arcpy.AddMessage("Uncertainty map provided, considering Th uncertainty")
		difference = np.sum((thbase - finalmap) ** 2 / uncertbase ** 2)
	    else:
		difference = np.sum((thbase - finalmap) ** 2)
	    arcpy.AddMessage("Computed a global difference of {}".format(difference))
	    
	    #Add the input raster minus the difference map
	    offsetroi = thbase - finalmap
	    derived = arcpy.NumPyArrayToRaster(offsetroi, lowerleft,
	                                       cellwidth,
	                                       cellheight,
	                                       None)	
	    if uncert != None:
		diffname = r'{}_Diff_Uncert'.format(outputname)
	    else:
		diffname = r'{}_Diff'.format(outputname)
	    
	    derivedpath = workspace + '\\' + diffname
	    derived.save(derivedpath)
	    res = arcpy.MakeRasterLayer_management(derivedpath, diffname + '_lyr')
	    rasterlyr = res.getOutput(0)
	    arcpy.mapping.AddLayer(df, rasterlyr, "TOP")		
    
	    #Chi-Squared
	    degrees_of_freedom = thbase.shape[0] * thbase.shape[1] - 1
	    diff2 = difference / float(degrees_of_freedom)
	    arcpy.AddMessage("Sum of the difference: {}\nDegrees of Freedom: {}\nChi-Squared = {}".format(difference, degrees_of_freedom,diff2))

	    #Append the chisquared and x-label to the axes
	    xlabels.append(outputname)
	    chisq.append(diff2)

	arcpy.AddMessage("Interpolating Observer Chi-Squared Values")
	#Create a table with data
	try:
	    arcpy.Delete_management("tempchisquared")
	except: pass
	
	table = arcpy.CreateTable_management(workspace, "tempchisquared")
	arcpy.AddField_management(table, "ID", "TEXT", field_length=20)
	arcpy.AddField_management(table, "ChiSq", "DOUBLE",
	                          field_precision=10, field_scale=5)
	
	#Perform the spline interpolation
	y = np.array(chisq)
	x = np.arange(len(xlabels))
	f = interp1d(x,y, kind='cubic', bounds_error=False)
	densex = np.linspace(0, len(xlabels) - 1, 10 * len(xlabels))
	newy = f(densex)
    
	cursor = arcpy.da.InsertCursor(table, ['ID', 'ChiSq'])
	for i, label in enumerate(densex):
	    if i % 10 == 0:
		label = xlabels[i / 10]
	    else:
		label = None
	    cursor.insertRow([label, newy[i]])
	
	arcpy.AddMessage("Interpolation complete, generating graph...")
	#Now generate the graph
	graph = arcpy.Graph()	
	graph_grf = r"C:\ArcGIS\Desktop10.2\GraphTemplates\default.tee"
	outgraphname = "Chi-Sq Testing for:"
	names = [i[1] for i in input_difference_maps]
	namelist = ",".join(names)
	outgraphname += namelist
	
	graph.addSeriesLineVertical(table, 'ChiSq', 'ID')
	
	graph.graphAxis[0] = "Map ID"
	graph.graphAxis[2] = "Chi-Squared"
	graph.graphPropsGeneral.title = "Comparison of Forward Modelled Th Abundances"

	arcpy.MakeGraph_management(graph_grf, graph, outgraphname)
	arcpy.AddMessage("GRaph made")
	#arcpy.AddMessage('Generated difference maps and computed Chi-squared.')
	#arcpy.AddMessage((zip(xlabels, chisq)))
        return



class ThoriumModelOnce(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Thorium Model"
        self.description = ""
        self.canRunInBackground = False

    def getParameterInfo(self):
        """Define parameter definitions"""

	#Workspace
	workspace = arcpy.Parameter(
	    displayName="Workspace",
	    name="workspace",
	    datatype="DEWorkspace",
	    parameterType="Required",
	    direction="Input")
	workspace.defaultEnvironmentName = "workspace"

	# Output Name
	outputname = arcpy.Parameter(
	    displayName="Output Name",
	    name="outputname",
	    datatype="GPString",
	    parameterType="Required",
	    direction="Input")

        # Thorium Raster
        thorium_raster = arcpy.Parameter(
            displayName="Thorium Model",
            name="thorium_raster",
            datatype="GPRasterLayer",
            parameterType="Required",
            direction="Input")

        # Input Features parameter
        in_features = arcpy.Parameter(
            displayName="Input ROI",
            name="in_features",
            datatype="GPFeatureLayer",
            parameterType="Required",
            direction="Input")
        in_features.filter.list = ["Polygon"]

        # Input Abundance Field
        abundance = arcpy.Parameter(
            displayName="Abundance Field",
            name="abundance",
            datatype="Field",
            parameterType="Required",
            direction="Input",
	    multiValue=True)
        abundance.parameterDependencies = [in_features.name]

        # Input Fill Value
        fill = arcpy.Parameter(
            displayName="Fill Value",
            name="fill",
            datatype="Double",
            parameterType="Required",
            direction="Input")
        fill.value = 3.88

        #Kappa
        padmode = arcpy.Parameter(
            displayName="Padding Mode",
            name="padmode",
            datatype="GPString",
            parameterType="Required",
            direction="Optional")
        padmode.filter.list = ['constant', 'edge', 'maximum', 'mean', 'reflect',
	                       'symmetric']
        padmode.value = 'constant'

        
        #Kappa
        kappa = arcpy.Parameter(
            displayName="Kappa",
            name="kappa",
            datatype="Double",
            parameterType="Required",
            direction="Optional")
        kappa.value = 0.625

        #Sigma
        sigma = arcpy.Parameter(
            displayName="Sigma",
            name="sigma",
            datatype="Double",
            parameterType="Required",
            direction="Optional")
        sigma.value = 22.5

        #Finite
        finite = arcpy.Parameter(
			displayName='Finite',
			name='finite',
			datatype='GPBoolean',
			parameterType='Optional',
			direction='Input')
        finite.value = True

        #Resolution
        resolution = arcpy.Parameter(
            displayName='Resolution',
            name='resolution',
            datatype='Long',
            parameterType='Optional',
            direction='Input')
        resolution.value = 8

	
        # Uncertainty Raster
        uncertraster = arcpy.Parameter(
            displayName="Thorium Uncertainty",
            name="uncertraster",
            datatype="GPRasterLayer",
            parameterType="Optional",
            direction="Input")

        params = [workspace, 
	          outputname, thorium_raster, in_features, abundance, fill,
	          padmode, kappa, sigma, finite, resolution, uncertraster]

        return params

    def isLicensed(self):
        """Set whether tool is licensed to execute."""
        return True

    def updateParameters(self, parameters):
        """Modify the values and properties of parameters before internal
        validation is performed.  This method is called whenever a parameter
        has been changed."""
        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter.  This method is called after internal validation."""
        return

    def execute(self, parameters, messages):
        """The source code of the tool."""
	workspace = parameters[0].valueAsText
	outputnames = parameters[1].valueAsText
        thorium = parameters[2].valueAsText
        roi = parameters[3].valueAsText
        abundances = parameters[4].valueAsText
        fill = parameters[5].value
        padmode = parameters[6].valueAsText
        kappa = parameters[7].value
        sigma = parameters[8].value
        finite = parameters[9].value
        resolution = parameters[10].value
	uncertraster = parameters[11].valueAsText

	try:
	    outputnames = outputnames.split(',')
	except:
	    outputnames = [outputnames]
	    
	try:
	    abundances = abundances.split(';')
	except:
	    abundances = [abundances]
	    
	for i, abundance in enumerate(abundances):
	    outputname = outputnames[i]
	    #Set the overwrite ENV variable to true
	    arcpy.env.overwriteOutput = True
	    
	    #Set the workspace
	    arcpy.env.workspace = workspace
	    
	    #Set the snapping environment and align the raster to the thorium base
	    arcpy.env.snapRaster = thorium
	    cellsize = arcpy.GetRasterProperties_management(thorium,
		                                            'CELLSIZEX').getOutput(0)
	    thorium_cellsize = cellsize
	    cellsize = float(cellsize) * (1.0 / resolution)
	    rasterized = '{}_rasterized'.format(outputname)
    
	    #Check for duplicate in_memory layer
	    deleteinmem(rasterized)
    
	    #Repair the geometries if necessary
	    arcpy.RepairGeometry_management(roi)
	    res = arcpy.PolygonToRaster_conversion(roi, abundance,
		                             rasterized,
		                             "MAXIMUM_AREA","",
		                             cellsize)
    
	    #Get into numpy
	    originalinput = arcpy.RasterToNumPyArray(rasterized)
    
	    originalinput[originalinput < -1e6] = fill
    
	    #Pad the array 2*resolution, i.e. pad 2 thorium resolution cells on all sides
	    pd = 2 * resolution
    
	    yshp, xshp = originalinput.shape
	    xshp += (pd * 2)
	    yshp += (pd * 2)
	    xremain = resolution - (xshp % resolution)
	    yremain = resolution - (yshp % resolution)
	    pdx = pd + xremain
	    pdy = pd + yremain
	    if str(padmode) != 'constant':
		inputArr = np.pad(originalinput, ((pdy,pd),(pd,pdx)), str(padmode))
	    else:
		inputArr = np.pad(originalinput, ((pdy,pd),(pd,pdx)), str(padmode),
		                   constant_values=(fill, fill))
	    
	    #Get the cell x,y coordinates of the rasterized layer
	    rasterdesc = arcpy.Describe(rasterized)
	    xmin = rasterdesc.Extent.XMin
	    ymin = rasterdesc.Extent.YMin
	    #Get cell information
	    cellheight = rasterdesc.MeanCellHeight
	    cellwidth = rasterdesc.MeanCellWidth
	    height = inputArr.shape[0]
	    width = inputArr.shape[1]
	    
	    #Because the Th layer is 1/2 deg. per pixel we can cheat
	    # and just do some simple math...
	    #Moving the origin since we padded all four edges
	    newxmin = xmin - 1.0
	    newymin = ymin - 1.0
	    lowerleft = arcpy.Point(newxmin, newymin)
    
	    #Get the padded raster, save to the workspace, and add to the map
	    padded = arcpy.NumPyArrayToRaster(inputArr,lowerleft, cellwidth,
		                              cellheight, None)
	    paddedname = '{}_padded'.format(outputname)
	    padded.save(workspace + '\\' + paddedname )
	    
	    #Convert the raster into raster layer for visualization
	    res = arcpy.MakeRasterLayer_management(padded, paddedname + 'lyr')
	    rasterlyr = res.getOutput(0)
    
	    #Add the rasterized layer to the data frame
	    mxd = arcpy.mapping.MapDocument("CURRENT")
	    df = arcpy.mapping.ListDataFrames(mxd, "*")[0]
	    arcpy.mapping.AddLayer(df, rasterlyr, "TOP")
	    arcpy.RefreshTOC()
	    arcpy.RefreshActiveView()
    
	    rasterinfo = {'xmin':newxmin, 'ymin':newymin, 'cellheight':cellheight,
		          'cellwidth':cellwidth, 'height':height, 
		          'width':width}
    
	    #Setup the output Coordinate system from the input
	    sr = rasterdesc.SpatialReference
	    arcpy.env.outputCoordinateSystem = sr
    
	    #Apply the kappasmoothing to the input
	    map0_fullres, finalmap_fullres = kappapreprocessing(inputArr, sigma, 
		                                                rasterinfo,
		                                                kappa, 
		                                                fill, 
		                                                cellsize, 
		                                                finite)
    
	    #Update the lower left back to the original origin , e.g. pre-padding.
	    lowerleft = arcpy.Point(xmin, ymin)
	    
	    #Add the derived products to the map in memory.
	    for k, mapproduct in {'map0_fullres':map0_fullres,
		         'finalmap_fullres':finalmap_fullres}.iteritems():
    
		#SEtup the map name
		mapname = "{}_{}".format(outputname, k)
		mapname_resampled = mapname.replace('fullres', 'resampled')
		deleteinmem(mapname)
		deleteinmem(mapname_resampled)
		
		#Remove the padded cells from the output.
		originalmapproduct = mapproduct[pd:-pd, pd:-pd]
		
		#Setup the spatial reference information for the output
		derived = arcpy.NumPyArrayToRaster(originalmapproduct, lowerleft,
		                                   cellwidth, cellheight, None)
		
		#Save the full resolution version to the workspace
		derived.save(workspace + '/{}_{}'.format(outputname, k))
		
		#Resample the kappa smoothed data back to the thorium resolution
		resampled = arcpy.Resample_management(derived, mapname_resampled,
		                             thorium_cellsize,'CUBIC')
    
		#This block add the high resolution results to the map
		res = arcpy.MakeRasterLayer_management(derived, mapname + '_lyr')
		rasterlyr = res.getOutput(0)
		arcpy.mapping.AddLayer(df, rasterlyr, "TOP")
    
		#This block adds the low resolution results to the map - resampled
		res = arcpy.MakeRasterLayer_management(mapname_resampled, mapname_resampled + '_lyr')
		rasterlyr = res.getOutput(0)
		arcpy.mapping.AddLayer(df, rasterlyr, "TOP")
		
		
		if k == 'finalmap_fullres':
		    finalmap = arcpy.RasterToNumPyArray(mapname_resampled)
		    arcpy.AddMessage("Comparing Derived to Observed...")
		    #Perform the Chi-Squared Comparison on the global dataset
		    #Extract the study area from the input raster and possibly the
		    # uncertainty raster
		    
		    thoriumroi = workspace + "/{}_thclip".format(outputname)
		    deleteinmem(thoriumroi)
		    referenceth = arcpy.Clip_management(thorium, "#", thoriumroi,
			                     rasterlyr)
		    
		    
		    #Add the reference layer to the map as a stand alone layer
		    res = arcpy.MakeRasterLayer_management(referenceth,
			                                  '{}_thclip_lyr'.format(outputname))
		    rasterlyr = res.getOutput(0)
		    arcpy.mapping.AddLayer(df, rasterlyr, "TOP")	
		    
		    if uncertraster != None:
			uncertroi = "{}_uncertclip".format(outputname)
			deleteinmem(uncertroi)		    
			reference_uncert = arcpy.Clip_management(uncertraster, "#",
			                              uncertroi, 
			                              rasterlyr)
		    
		    throi = arcpy.RasterToNumPyArray(thoriumroi)		
		    
		    #Compute the global difference
		    #If uncertainty is provided compute the diff using that as well		
		    if uncertraster != None:
			arcpy.AddMessage("Uncertainty map provided, considering Th uncertainty")
			uncertroi = arcpy.RasterToNumPyArray(uncertroi)
			arcpy.AddMessage(uncertroi.shape)
			difference = np.sum((throi - finalmap) ** 2 / uncertroi ** 2)
		    else:
			difference = np.sum((throi - finalmap) ** 2)
		    arcpy.AddMessage("Computed a global difference of {}".format(difference))
		    
		    #Add the input raster minus the difference map
		    offsetroi = throi - finalmap
		    derived = arcpy.NumPyArrayToRaster(offsetroi, lowerleft,
			                               cellwidth * resolution,
			                               cellheight * resolution)	
		    if uncertraster != None:
			diffname = r'{}_Diff_Uncert'.format(outputname)
		    else:
			diffname = r'{}_Diff'.format(outputname)
		    
		    derivedpath = workspace + '\\' + diffname
		    derived.save(derivedpath)
		    res = arcpy.MakeRasterLayer_management(derivedpath, diffname + '_lyr')	    

		    rasterlyr = res.getOutput(0)
		    arcpy.mapping.AddLayer(df, rasterlyr, "TOP")		
	    
		    #Chi-Squared
		    degrees_of_freedom = throi.shape[0] * throi.shape[1] - 1
		    diff2 = difference / float(degrees_of_freedom)
		    arcpy.AddMessage("Sum of the difference: {}\nDegrees of Freedom: {}\nChi-Squared = {}".format(difference, degrees_of_freedom,diff2))
		
	    
        return