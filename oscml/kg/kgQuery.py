from oscml.kg.kgGateway import jpsBaseLibGW

jpsBaseLib_view = jpsBaseLibGW.createModuleView()
jpsBaseLibGW.importPackages(jpsBaseLib_view,"uk.ac.cam.cares.jps.base.query.*")

def queryKG(sparqlEndPoint=None, queryStr=None):
    queryStr = "\n".join(queryStr)

    KGClient = jpsBaseLib_view.RemoteKnowledgeBaseClient(sparqlEndPoint)
    response = KGClient.executeQuery(queryStr)
    response = str(response)

    return response