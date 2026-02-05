#!/bin/bash


# Upload to Zenodo Script
# This script uploads compressed data and models to Zenodo using the REST API.
# Author: zhangshd
# Date: July 3, 2025


# Configuration - Token Security
# Try to get token from environment variable first, then from file, then prompt user
if [ -n "$ZENODO_TOKEN" ]; then
    echo "Using token from environment variable"
elif [ -f ~/.zenodo_token ]; then
    ZENODO_TOKEN=$(cat ~/.zenodo_token)
    echo "Using token from ~/.zenodo_token file"
else
    echo "No token found in environment or file"
    echo "You can set token by:"
    echo "1. Export environment variable: export ZENODO_TOKEN='your_token'"
    echo "2. Create file: echo 'your_token' > ~/.zenodo_token && chmod 600 ~/.zenodo_token"
    echo "3. Enter token interactively below"
    read -s -p "Enter Zenodo Token: " ZENODO_TOKEN
    echo
fi

# Validate token is set
if [ -z "$ZENODO_TOKEN" ]; then
    echo "Error: No Zenodo token provided"
    exit 1
fi

# Zenodo environments
ZENODO_PROD_URL="https://zenodo.org/api/deposit/depositions"          # Production (official)
ZENODO_SANDBOX_URL="https://sandbox.zenodo.org/api/deposit/depositions"  # Sandbox (testing)

# Default to production, but you can change this
USE_SANDBOX=false  # Set to true for testing, false for production

if [ "$USE_SANDBOX" = true ]; then
    ZENODO_URL="$ZENODO_SANDBOX_URL"
    echo "Using Zenodo SANDBOX environment (testing only)"
else
    ZENODO_URL="$ZENODO_PROD_URL"
    echo "Using Zenodo PRODUCTION environment (official)"
fi

# Configuration for Version 2 (use the already created draft deposition)
EXISTING_DEPOSITION_ID="18478768"   # Draft deposition ID already created
EXISTING_DOI=""                     # Leave empty since we're using existing draft
CONCEPT_DOI=""                      # Set if referencing related work

# Function to create a new deposition
create_deposition() {
    echo "Creating new deposition..."
    
    response=$(curl -s -w "HTTP_CODE:%{http_code}" \
        -X POST \
        -H "Authorization: Bearer $ZENODO_TOKEN" \
        -H "Content-Type: application/json" \
        "$ZENODO_URL" \
        -d '{}')
    
    http_code=$(echo "$response" | grep -o 'HTTP_CODE:[0-9]*' | cut -d: -f2)
    response_body=$(echo "$response" | sed 's/HTTP_CODE:[0-9]*$//')
    
    echo "HTTP Status Code: $http_code"
    
    if [ "$http_code" != "201" ]; then
        echo "Error: Failed to create deposition (HTTP $http_code)"
        echo "Response: $response_body"
        return 1
    fi
    
    # Extract deposition ID from response
    deposition_id=$(echo "$response_body" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    if 'id' in data:
        print(data['id'])
    else:
        print('ERROR: No ID in response')
        print(f'Available keys: {list(data.keys())}', file=sys.stderr)
        exit(1)
except json.JSONDecodeError as e:
    print('ERROR: Invalid JSON response')
    print(f'JSON Error: {e}', file=sys.stderr)
    exit(1)
except Exception as e:
    print('ERROR: Unexpected error')
    print(f'Error: {e}', file=sys.stderr)
    exit(1)
")
    
    if [[ "$deposition_id" == ERROR* ]]; then
        echo "Error: Failed to create deposition"
        echo "Response: $response_body"
        return 1
    fi
    
    echo "Created deposition with ID: $deposition_id"
    echo "$deposition_id"
}

# Function to upload a file
upload_file() {
    local deposition_id=$1
    local file_path=$2
    local filename=$(basename "$file_path")
    
    echo "Uploading $filename..."
    
    # Get bucket URL
    echo "Getting bucket URL for deposition $deposition_id..."
    
    response=$(curl -s \
        -H "Authorization: Bearer $ZENODO_TOKEN" \
        "$ZENODO_URL/$deposition_id")
    
    # Debug: show response
    echo "API Response for getting bucket URL:"
    echo "$response" | head -5
    
    bucket_url=$(echo "$response" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    if 'links' in data and 'bucket' in data['links']:
        print(data['links']['bucket'])
    else:
        print('ERROR: No bucket link found')
        print(f'Available keys: {list(data.keys())}', file=sys.stderr)
        if 'links' in data:
            print(f'Available links: {list(data[\"links\"].keys())}', file=sys.stderr)
        exit(1)
except json.JSONDecodeError as e:
    print('ERROR: Invalid JSON response')
    print(f'JSON Error: {e}', file=sys.stderr)
    exit(1)
except Exception as e:
    print('ERROR: Unexpected error')
    print(f'Error: {e}', file=sys.stderr)
    exit(1)
")
    
    if [[ "$bucket_url" == ERROR* ]]; then
        echo "Error: Could not get bucket URL"
        echo "Response was: $response"
        return 1
    fi
    
    echo "Bucket URL: $bucket_url"
    
    # Upload file
    response=$(curl -X PUT \
        -H "Authorization: Bearer $ZENODO_TOKEN" \
        -H "Content-Type: application/octet-stream" \
        --upload-file "$file_path" \
        "$bucket_url/$filename")
    
    if [ $? -eq 0 ]; then
        echo "Successfully uploaded $filename"
        return 0
    else
        echo "Error uploading $filename"
        return 1
    fi
}

# Function to create new version from existing DOI
create_new_version() {
    local existing_doi=$1
    
    echo "Creating new version from existing DOI: $existing_doi" >&2
    
    # Extract record ID from DOI
    local record_id=$(echo "$existing_doi" | grep -o '[0-9]*$')
    
    if [ -z "$record_id" ]; then
        echo "Error: Could not extract record ID from DOI: $existing_doi" >&2
        return 1
    fi
    
    # Create new version
    response=$(curl -s -X POST \
        -H "Authorization: Bearer $ZENODO_TOKEN" \
        "$ZENODO_URL/$record_id/actions/newversion")
    
    # Extract new deposition ID from response
    new_deposition_id=$(echo "$response" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    # Get the latest draft from the new version
    print(data['links']['latest_draft'].split('/')[-1])
except Exception as e:
    print('ERROR: Could not parse new version response', file=sys.stderr)
    print(f'Error: {e}', file=sys.stderr)
    print(f'Response: {data}', file=sys.stderr)
    exit(1)
")
    
    if [ $? -ne 0 ] || [ -z "$new_deposition_id" ]; then
        echo "Error: Failed to create new version" >&2
        echo "Response: $response" >&2
        return 1
    fi
    
    echo "Created new version with deposition ID: $new_deposition_id" >&2
    echo "$new_deposition_id"
}

# Function to get existing deposition info
get_deposition_info() {
    local deposition_id=$1
    
    echo "Getting deposition info for ID: $deposition_id"
    
    response=$(curl -s \
        -H "Authorization: Bearer $ZENODO_TOKEN" \
        "$ZENODO_URL/$deposition_id")
    
    echo "Current deposition info:"
    echo "$response" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(f'Title: {data.get(\"title\", \"N/A\")}')
    print(f'DOI: {data.get(\"doi\", \"N/A\")}')
    print(f'State: {data.get(\"state\", \"N/A\")}')
    print(f'Submitted: {data.get(\"submitted\", \"N/A\")}')
    print(f'Files: {len(data.get(\"files\", []))}')
except Exception as e:
    print('Error parsing deposition info')
    print(f'Raw response: {sys.stdin.read()}')
"
}

# Function to update metadata with related DOI information
update_metadata_with_doi() {
    local deposition_id=$1
    local related_doi=$2
    
    echo "Updating metadata with related DOI information..."
    
    # Enhanced metadata with DOI relations
    metadata='{
        "metadata": {
            "title": "MAPP: A Generic Model for Mixture Adsorption Property Prediction - Data and Models",
            "upload_type": "dataset",
            "description": "This dataset contains the compressed data and trained models for the MAPP project, including CGCNN and MOFTransformer models for predicting gas adsorption properties in Metal-Organic Frameworks (MOFs).",
            "creators": [
                {
                    "name": "Zhang, Shengda",
                    "affiliation": "Hong Kong University of Science and Technology (Guangzhou)"
                }
            ],
            "keywords": ["MOF", "Machine Learning", "Gas Adsorption", "CGCNN", "MOFTransformer"],
            "access_right": "open",
            "license": "cc-by-4.0",
            "version": "1.0"'
    
    # Add related identifiers if DOI is provided
    if [ -n "$related_doi" ]; then
        metadata+=',
            "related_identifiers": [
                {
                    "identifier": "'$related_doi'",
                    "relation": "isVersionOf",
                    "resource_type": "dataset"
                }
            ]'
    fi
    
    metadata+='
        }
    }'
    
    response=$(curl -X PUT \
        -H "Authorization: Bearer $ZENODO_TOKEN" \
        -H "Content-Type: application/json" \
        "$ZENODO_URL/$deposition_id" \
        -d "$metadata")
    
    echo "Metadata updated with DOI relations"
}

# Function to update metadata for Version 2
update_metadata() {
    local deposition_id=$1
    
    echo "Updating metadata for Version 2.0..."
    
    # Version 2.0 description
    description='This repository contains the training data and trained model files used in our work titled "A Generic Model for Mixture Adsorption Property Prediction (MAPP): Toward Efficient Discovery of Metal-Organic Frameworks for CO2 Capture". The goal of this study is to predict CO₂/N₂ adsorption in MOFs under various conditions (e.g. pressure and gas composition). The training data were curated from GCMC simulation using RASPA 2.0.47, and the model was trained using Pytorch 1.13.1.\n\n## Version History\n\n**Version 2.0** (Current): Updated models and extended data\n- Added MAPP-V4 models (ExTransformerV4 architecture) with improved performance\n- Added CGCNN baseline models for comparison\n- Added pure component model for IAST predictions\n- Added extended training data with original absolute loading and excess loading data, and multiple transformation formats (log10/symlog10/arcsinh)\n- Added experimental MOF validation data (IRMOF-1, UiO-66, MIL-53(Al), CALF-20)\n\n**Version 1.0**: Initial release with V3 models (available via version history)\n\n## Included Files (Version 2.0)\n\n**Data Files:**\n- cifs_graphs_grids.tar.gz: Processed MOFs CIF, CGCNN graph, and energy grid files (inherited from v1)\n- mof_split_val1000_test1000_seed0_org_csv.zip: Labeled data with multiple adsorption representations, MOF-based split\n- mof_cluster_split_val1_test3_seed0_org_csv.zip: Labeled data with multiple adsorption representations, cluster-based split\n- isotherm_extra_test.tar.gz: Extra test set isotherm data from GCMC simulation\n- processed_data.tar.gz: Processed training data (id_condition_ads_qst_org_all.csv)\n- exp_mof_data.tar.gz: Experimental MOF validation data and simulation data (5 MOFs)\n\n**MAPP-V4 Models (MOFTransformer-based):**\n- model_MAPP_GMOF_v4.tar.gz: MAPP model trained on MOF-based split\n- model_MAPP_GCluster_v4.tar.gz: MAPP model trained on cluster-based split\n- model_MAPPPure.tar.gz: Pure component model for IAST predictions\n\n**CGCNN Baseline Models:**\n- model_CGCNN_GMOF.tar.gz: CGCNN model trained on MOF-based split\n- model_CGCNN_GCluster.tar.gz: CGCNN model trained on cluster-based split'
    
    # Create metadata JSON
    metadata='{
        "metadata": {
            "title": "MAPP: A Generic Model for Mixture Adsorption Property Prediction - Data and Models",
            "upload_type": "dataset",
            "publication_date": "'$(date +%Y-%m-%d)'",
            "description": "'"$description"'",
            "creators": [
                {
                    "name": "Zhang, Shengda",
                    "affiliation": "Hong Kong University of Science and Technology (Guangzhou)"
                }
            ],
            "keywords": ["MOF", "Machine Learning", "Gas Adsorption", "CGCNN", "MOFTransformer", "CO2 Capture"],
            "access_right": "open",
            "license": "cc-by-4.0",
            "version": "2.0"
        }
    }'
    
    response=$(curl -X PUT \
        -H "Authorization: Bearer $ZENODO_TOKEN" \
        -H "Content-Type: application/json" \
        "$ZENODO_URL/$deposition_id" \
        -d "$metadata")
    
    echo "Metadata updated for Version 2.0"
}

# Function to publish deposition
publish_deposition() {
    local deposition_id=$1
    
    echo "Publishing deposition..."
    
    response=$(curl -X POST \
        -H "Authorization: Bearer $ZENODO_TOKEN" \
        "$ZENODO_URL/$deposition_id/actions/publish")
    
    # Extract DOI from response
    doi=$(echo "$response" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(data['doi'])
except:
    print('ERROR')
    exit(1)
")
    
    if [ "$doi" = "ERROR" ]; then
        echo "Error: Could not publish deposition"
        echo "Response: $response"
        return 1
    fi
    
    echo "Successfully published! DOI: $doi"
    echo "Zenodo URL: https://zenodo.org/record/$deposition_id"
}

# Function to test token validity
test_token() {
    echo "Testing token validity..."
    
    response=$(curl -s -w "HTTP_CODE:%{http_code}" \
        -H "Authorization: Bearer $ZENODO_TOKEN" \
        "$ZENODO_URL")
    
    http_code=$(echo "$response" | grep -o 'HTTP_CODE:[0-9]*' | cut -d: -f2)
    response_body=$(echo "$response" | sed 's/HTTP_CODE:[0-9]*$//')
    
    echo "HTTP Status Code: $http_code"
    
    if [ "$http_code" = "200" ]; then
        echo "Token is valid"
        return 0
    elif [ "$http_code" = "401" ]; then
        echo "Error: Invalid token or token expired"
        echo "Please check your token at: $ZENODO_URL/../account/settings/applications/"
        return 1
    elif [ "$http_code" = "403" ]; then
        echo "Error: Token valid but insufficient permissions"
        echo "Please ensure token has 'deposit:write' and 'deposit:actions' scopes"
        return 1
    else
        echo "Error: Unexpected HTTP status code: $http_code"
        echo "Response: $response_body"
        return 1
    fi
}

# Main execution
main() {
    # Check if token is set (removed old check since we now handle token differently)
    
    # Test token validity
    test_token
    if [ $? -ne 0 ]; then
        echo "Error: Invalid token. Please fix the token issue before proceeding."
        exit 1
    fi
    
    # Determine how to create deposition
    if [ -n "$EXISTING_DEPOSITION_ID" ]; then
        echo "Using existing deposition ID: $EXISTING_DEPOSITION_ID"
        deposition_id="$EXISTING_DEPOSITION_ID"
    elif [ -n "$EXISTING_DOI" ]; then
        echo "Creating new version from existing DOI: $EXISTING_DOI"
        deposition_id=$(create_new_version "$EXISTING_DOI")
    else
        echo "Creating new deposition..."
        deposition_id=$(create_deposition)
    fi
    
    if [ -z "$deposition_id" ]; then
        echo "Error: Failed to create deposition"
        exit 1
    fi
    
    # List of files to upload for Version 2 (only new files, inherited files come from Version 1)
    files=(
        # Data splits (_org versions)
        "CGCNN_MT/data/ddmof/mof_cluster_split_val1_test3_seed0_org_csv.zip"
        "CGCNN_MT/data/ddmof/mof_split_val1000_test1000_seed0_org_csv.zip"
        
        # MAPP V4 Models (MOFTransformer-based)
        "MOFTransformer/logs/ads_co2_n2_org_v4_seed42_extranformerv4_from_pmtransformer/model_MAPP_GMOF_v4.tar.gz"
        "MOFTransformer/logs/ads_co2_n2_org_v4_seed42_extranformerv4_from_pmtransformer/model_MAPP_GCluster_v4.tar.gz"
        "MOFTransformer/logs/ads_co2_n2_pure_v4_seed42_extranformerv4_from_pmtransformer/model_MAPPPure.tar.gz"
        
        # CGCNN Baseline Models
        "CGCNN_MT/logs/SymlogAbsLoadingCO2_SymlogAbsLoadingN2_seed42_cgcnn_langmuir/model_CGCNN_GMOF.tar.gz"
        "CGCNN_MT/logs/SymlogAbsLoadingCO2_SymlogAbsLoadingN2_seed42_cgcnn_langmuir/model_CGCNN_GCluster.tar.gz"
        
        # Additional data files
        "CGCNN_MT/data/exp_mof_data.tar.gz"
        "CGCNN_MT/data/ddmof/isotherm_extra_test.tar.gz"
        "CGCNN_MT/data/ddmof/processed_data.tar.gz"
    )
    
    # Upload files
    upload_success=0
    for file in "${files[@]}"; do
        if [ -f "$file" ]; then
            if upload_file "$deposition_id" "$file"; then
                ((upload_success++))
            fi
        else
            echo "Warning: File not found: $file"
        fi
    done
    
    if [ $upload_success -eq 0 ]; then
        echo "Error: No files were uploaded successfully"
        exit 1
    fi
    
    # Update metadata (with DOI relations if applicable)
    if [ -n "$CONCEPT_DOI" ]; then
        update_metadata_with_doi "$deposition_id" "$CONCEPT_DOI"
    else
        update_metadata "$deposition_id"
    fi
    
    # Ask user if they want to publish
    read -p "Do you want to publish the deposition now? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        publish_deposition "$deposition_id"
    else
        echo "Deposition created but not published. You can publish it later from:"
        echo "https://zenodo.org/deposit/$deposition_id"
    fi
}

# Run main function
main "$@"
