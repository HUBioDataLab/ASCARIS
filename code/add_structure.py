import re
import time
import json
import zlib
from xml.etree import ElementTree
from urllib.parse import urlparse, parse_qs, urlencode
import requests
import unipressed
from requests.adapters import HTTPAdapter, Retry
from unipressed import IdMappingClient
"""
## Code adapted from UniProt documentation.
def get_pdb_ids_2(protein_id):
    POLLING_INTERVAL = 5
    API_URL = "https://rest.uniprot.org"

    retries = Retry(total=5, backoff_factor=0.25, status_forcelist=[500, 502, 503, 504])
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retries))

    def check_response(response):
        try:
            response.raise_for_status()
        except requests.HTTPError:
            print(response.json())
            raise

    def submit_id_mapping(from_db, to_db, ids):
        request = requests.post(
            f"{API_URL}/idmapping/run",
            data={"from": from_db, "to": to_db, "ids": ids},
        )
        check_response(request)
        if check_response != None:
        	return request.json()["jobId"]
        else:
        	return None

    def get_next_link(headers):
        re_next_link = re.compile(r'<(.+)>; rel="next"')
        if "Link" in headers:
            match = re_next_link.match(headers["Link"])
            if match:
                return match.group(1)

    def check_id_mapping_results_ready(job_id):
        print('entered')
        while True:
            print('True')
            print('HR-1')
            try:
                request = session.get(f"{API_URL}/idmapping/status/{job_id}")
            except requests.exceptions.RetryError:
                print('eneted')
                request = None
            print('HR0-22')
            check_response(request)
            j = request.json()
            print('HR0')
            try:
                print('HR1')
                if "jobStatus" in j:
                    print('HR2')
                    if j["jobStatus"] == "RUNNING":
                        print(f"Retrying in {POLLING_INTERVAL}s")
                        time.sleep(POLLING_INTERVAL)
                    else:
                        print('HR3')
                        raise Exception(j["jobStatus"])
            except:
                print('HR4')
                requests.exceptions.RetryError
            else:
                print('HR4')
                return bool(j["results"] or j["failedIds"])

    def get_batch(batch_response, file_format, compressed):
        batch_url = get_next_link(batch_response.headers)
        while batch_url:
            batch_response = session.get(batch_url)
            batch_response.raise_for_status()
            yield decode_results(batch_response, file_format, compressed)
            batch_url = get_next_link(batch_response.headers)

    def combine_batches(all_results, batch_results, file_format):
        if file_format == "json":
            for key in ("results", "failedIds"):
                if key in batch_results and batch_results[key]:
                    all_results[key] += batch_results[key]
        elif file_format == "tsv":
            return all_results + batch_results[1:]
        else:
            return all_results + batch_results
        return all_results

    def get_id_mapping_results_link(job_id):
        url = f"{API_URL}/idmapping/details/{job_id}"

        request = session.get(url)
        check_response(request)
        return request.json()["redirectURL"]

    def decode_results(response, file_format, compressed):
        if compressed:
            decompressed = zlib.decompress(response.content, 16 + zlib.MAX_WBITS)
            if file_format == "json":
                j = json.loads(decompressed.decode("utf-8"))
                return j
            elif file_format == "tsv":
                return [line for line in decompressed.decode("utf-8").split("\n") if line]
            elif file_format == "xlsx":
                return [decompressed]
            elif file_format == "xml":
                return [decompressed.decode("utf-8")]
            else:
                return decompressed.decode("utf-8")
        elif file_format == "json":
            return response.json()
        elif file_format == "tsv":
            return [line for line in response.text.split("\n") if line]
        elif file_format == "xlsx":
            return [response.content]
        elif file_format == "xml":
            return [response.text]
        return response.text

    def get_xml_namespace(element):
        m = re.match(r"\{(.*)\}", element.tag)
        return m.groups()[0] if m else ""

    def merge_xml_results(xml_results):
        merged_root = ElementTree.fromstring(xml_results[0])
        for result in xml_results[1:]:
            root = ElementTree.fromstring(result)
            for child in root.findall("{http://uniprot.org/uniprot}entry"):
                merged_root.insert(-1, child)
        ElementTree.register_namespace("", get_xml_namespace(merged_root[0]))
        return ElementTree.tostring(merged_root, encoding="utf-8", xml_declaration=True)


    def get_id_mapping_results_search(url):
        parsed = urlparse(url)
        query = parse_qs(parsed.query)
        file_format = query["format"][0] if "format" in query else "json"
        if "size" in query:
            size = int(query["size"][0])
        else:
            size = 500
            query["size"] = size
        compressed = (
            query["compressed"][0].lower() == "true" if "compressed" in query else False
        )
        parsed = parsed._replace(query=urlencode(query, doseq=True))
        url = parsed.geturl()
        request = session.get(url)
        check_response(request)
        results = decode_results(request, file_format, compressed)
        total = int(request.headers["x-total-results"])
        for i, batch in enumerate(get_batch(request, file_format, compressed), 1):
            results = combine_batches(results, batch, file_format)
        if file_format == "xml":
            return merge_xml_results(results)
        return results


    job_id = submit_id_mapping(
        from_db="UniProtKB_AC-ID", to_db="PDB", ids=protein_id
    )
    print('skhfkh')
    print(submit_id_mapping(
        from_db="UniProtKB_AC-ID", to_db="PDB", ids=protein_id
    ))
    print('nor', check_id_mapping_results_ready(job_id))
    if check_id_mapping_results_ready(job_id):
        link = get_id_mapping_results_link(job_id)
        results = get_id_mapping_results_search(link)
        return [i['to'] for i in results['results']]
    else:
        print('no i am here')
        return None
def get_pdb_ids(protein_id):
    try:
        request = IdMappingClient.submit(
            source="UniProtKB_AC-ID", dest="PDB", ids={protein_id})

        try:
            pdb_list = list(request.each_result())
            time.sleep(1)
            return [i['to'] for i in pdb_list]
        except unipressed.id_mapping.core.IdMappingError:
            print('I AM HERE 1')
            get_pdb_ids_2(protein_id)
    except requests.exceptions.HTTPError:
        print('I AM HERE 2')
        get_pdb_ids_2(protein_id)
    except KeyError:
        print('I AM HERE 3')
        get_pdb_ids_2(protein_id)
"""

def get_pdb_ids(protein_id):
    try:
        request = IdMappingClient.submit(
            source="UniProtKB_AC-ID", dest="PDB", ids={protein_id})
        pdb_list = list(request.each_result())
        return [i['to'] for i in pdb_list]
    except requests.exceptions.HTTPError:
        return  []
    except unipressed.id_mapping.core.IdMappingError:
        print('IdMappingError caused by UniProt API service, please try later.')
        return  []
    except KeyError:
        return  []