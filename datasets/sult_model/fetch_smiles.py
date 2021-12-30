# Script to fetch smiles using chembl id 
# Source: https://www.ebi.ac.uk/chembl/ws

# For monkey patching (necessary?)
# import gevent.monkey
# gevent.monkey.patch_all()
# from requests.packages.urllib3.util.ssl_ import create_urllib3_context
# create_urllib3_context()

from chembl_webresource_client.new_client import new_client
molecule = new_client.molecule

def get_smiles(_id):
    mol = molecule.get(_id)
    return mol["molecule_structures"]["canonical_smiles"]

print(get_smiles("CHEMBL300797"))
