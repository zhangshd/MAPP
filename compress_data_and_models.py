"""
Compress Data and Models Script for Zenodo Version 2
This script compresses and packages the project data and models for Zenodo upload.
Author: zhangshd
Date: February 4, 2025
"""

import os
import sys
import tarfile
import zipfile
import shutil
from pathlib import Path
from typing import List, Optional, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataModelCompressor:
    """
    A class to compress and package project data and models for Zenodo Version 2.
    """
    
    def __init__(self, project_root: str):
        """
        Initialize the compressor with project root path.
        
        Args:
            project_root: The root directory of the project
        """
        self.project_root = Path(project_root).resolve()
        self.compression_info = []
        
        # Validate project root exists
        if not self.project_root.exists():
            raise ValueError(f"Project root does not exist: {self.project_root}")
    
    def compress_directory_tar(self, source_dir: Path, output_path: Path, 
                               exclude_patterns: Optional[List[str]] = None) -> bool:
        """
        Compress a directory using tar.gz format.
        
        Args:
            source_dir: Source directory to compress
            output_path: Output tar.gz file path
            exclude_patterns: List of patterns to exclude from compression
            
        Returns:
            True if compression successful, False otherwise
        """
        try:
            if not source_dir.exists():
                logger.error(f"Source directory does not exist: {source_dir}")
                return False
                
            logger.info(f"Compressing {source_dir} to {output_path}")
            
            # Create output directory if it doesn't exist
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with tarfile.open(output_path, 'w:gz') as tar:
                tar.add(source_dir, arcname=source_dir.name, 
                       filter=self._create_tar_filter(exclude_patterns))
            
            # Get compression info
            original_size = self._get_directory_size(source_dir)
            compressed_size = output_path.stat().st_size
            compression_ratio = (1 - compressed_size / original_size) * 100 if original_size > 0 else 0
            
            self.compression_info.append({
                'source': str(source_dir),
                'output': str(output_path),
                'original_size_mb': original_size / (1024 * 1024),
                'compressed_size_mb': compressed_size / (1024 * 1024),
                'compression_ratio': compression_ratio
            })
            
            logger.info(f"Successfully compressed {source_dir.name}")
            logger.info(f"Original size: {original_size / (1024 * 1024):.2f} MB")
            logger.info(f"Compressed size: {compressed_size / (1024 * 1024):.2f} MB")
            logger.info(f"Compression ratio: {compression_ratio:.1f}%")
            
            return True
            
        except Exception as e:
            logger.error(f"Error compressing {source_dir}: {str(e)}")
            return False
    
    def compress_file_tar(self, source_file: Path, output_path: Path) -> bool:
        """
        Compress a single file using tar.gz format.
        
        Args:
            source_file: Source file to compress
            output_path: Output tar.gz file path
            
        Returns:
            True if compression successful, False otherwise
        """
        try:
            if not source_file.exists():
                logger.error(f"Source file does not exist: {source_file}")
                return False
                
            logger.info(f"Compressing {source_file} to {output_path}")
            
            # Create output directory if it doesn't exist
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with tarfile.open(output_path, 'w:gz') as tar:
                tar.add(source_file, arcname=source_file.name)
            
            # Get compression info
            original_size = source_file.stat().st_size
            compressed_size = output_path.stat().st_size
            compression_ratio = (1 - compressed_size / original_size) * 100 if original_size > 0 else 0
            
            self.compression_info.append({
                'source': str(source_file),
                'output': str(output_path),
                'original_size_mb': original_size / (1024 * 1024),
                'compressed_size_mb': compressed_size / (1024 * 1024),
                'compression_ratio': compression_ratio
            })
            
            logger.info(f"Successfully compressed {source_file.name}")
            logger.info(f"Original size: {original_size / (1024 * 1024):.2f} MB")
            logger.info(f"Compressed size: {compressed_size / (1024 * 1024):.2f} MB")
            logger.info(f"Compression ratio: {compression_ratio:.1f}%")
            
            return True
            
        except Exception as e:
            logger.error(f"Error compressing {source_file}: {str(e)}")
            return False
    
    def compress_files_zip(self, source_dir: Path, output_path: Path, 
                          file_pattern: str = "*.csv") -> bool:
        """
        Compress specific files in a directory using zip format.
        
        Args:
            source_dir: Source directory containing files to compress
            output_path: Output zip file path
            file_pattern: Pattern to match files (default: "*.csv")
            
        Returns:
            True if compression successful, False otherwise
        """
        try:
            if not source_dir.exists():
                logger.error(f"Source directory does not exist: {source_dir}")
                return False
                
            # Find matching files
            matching_files = list(source_dir.glob(file_pattern))
            
            if not matching_files:
                logger.warning(f"No files matching pattern '{file_pattern}' found in {source_dir}")
                return False
                
            logger.info(f"Compressing {len(matching_files)} files from {source_dir} to {output_path}")
            
            # Create output directory if it doesn't exist
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            original_size = 0
            with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                for file_path in matching_files:
                    arc_name = file_path.name
                    zip_file.write(file_path, arc_name)
                    original_size += file_path.stat().st_size
                    logger.info(f"Added {file_path.name} to archive")
            
            # Get compression info
            compressed_size = output_path.stat().st_size
            compression_ratio = (1 - compressed_size / original_size) * 100 if original_size > 0 else 0
            
            self.compression_info.append({
                'source': str(source_dir),
                'output': str(output_path),
                'original_size_mb': original_size / (1024 * 1024),
                'compressed_size_mb': compressed_size / (1024 * 1024),
                'compression_ratio': compression_ratio,
                'file_count': len(matching_files)
            })
            
            logger.info(f"Successfully compressed {len(matching_files)} files")
            logger.info(f"Original size: {original_size / (1024 * 1024):.2f} MB")
            logger.info(f"Compressed size: {compressed_size / (1024 * 1024):.2f} MB")
            logger.info(f"Compression ratio: {compression_ratio:.1f}%")
            
            return True
            
        except Exception as e:
            logger.error(f"Error compressing files in {source_dir}: {str(e)}")
            return False
    
    def _create_tar_filter(self, exclude_patterns: Optional[List[str]] = None):
        """Create a filter function for tar compression."""
        def tar_filter(tarinfo):
            if exclude_patterns:
                for pattern in exclude_patterns:
                    if pattern in tarinfo.name:
                        return None
            return tarinfo
        
        return tar_filter if exclude_patterns else None
    
    def _get_directory_size(self, directory: Path) -> int:
        """Calculate the total size of a directory."""
        total_size = 0
        for path in directory.rglob('*'):
            if path.is_file():
                total_size += path.stat().st_size
        return total_size
    
    def compress_all_targets_v2(self) -> bool:
        """
        Compress all target directories and files for Zenodo Version 2.
        
        Returns:
            True if all compressions successful, False otherwise
        """
        success_count = 0
        total_tasks = 10
        
        logger.info("=" * 80)
        logger.info("Starting compression for Zenodo Version 2...")
        logger.info("=" * 80)
        
        # 1. Compress CSV files in mof_cluster_split_val1_test3_seed0_org
        logger.info("\n[1/10] Compressing mof_cluster_split_val1_test3_seed0_org CSV files...")
        cluster_split_org_dir = self.project_root / "CGCNN_MT" / "data" / "ddmof" / "mof_cluster_split_val1_test3_seed0_org"
        cluster_split_org_output = self.project_root / "CGCNN_MT" / "data" / "ddmof" / "mof_cluster_split_val1_test3_seed0_org_csv.zip"
        
        if self.compress_files_zip(cluster_split_org_dir, cluster_split_org_output, "*.csv"):
            success_count += 1
        
        # 2. Compress CSV files in mof_split_val1000_test1000_seed0_org
        logger.info("\n[2/10] Compressing mof_split_val1000_test1000_seed0_org CSV files...")
        split_org_dir = self.project_root / "CGCNN_MT" / "data" / "ddmof" / "mof_split_val1000_test1000_seed0_org"
        split_org_output = self.project_root / "CGCNN_MT" / "data" / "ddmof" / "mof_split_val1000_test1000_seed0_org_csv.zip"
        
        if self.compress_files_zip(split_org_dir, split_org_output, "*.csv"):
            success_count += 1
        
        # 3. Compress MOFTransformer V4 GMOF model (version_14)
        logger.info("\n[3/10] Compressing MAPP GMOF V4 model...")
        mapp_gmof_v4_dir = self.project_root / "MOFTransformer" / "logs" / "ads_co2_n2_org_v4_seed42_extranformerv4_from_pmtransformer" / "version_14"
        mapp_gmof_v4_output = self.project_root / "MOFTransformer" / "logs" / "ads_co2_n2_org_v4_seed42_extranformerv4_from_pmtransformer" / "model_MAPP_GMOF_v4.tar.gz"
        
        if self.compress_directory_tar(mapp_gmof_v4_dir, mapp_gmof_v4_output):
            success_count += 1
        
        # 4. Compress MOFTransformer V4 GCluster model (version_12)
        logger.info("\n[4/10] Compressing MAPP GCluster V4 model...")
        mapp_gcluster_v4_dir = self.project_root / "MOFTransformer" / "logs" / "ads_co2_n2_org_v4_seed42_extranformerv4_from_pmtransformer" / "version_12"
        mapp_gcluster_v4_output = self.project_root / "MOFTransformer" / "logs" / "ads_co2_n2_org_v4_seed42_extranformerv4_from_pmtransformer" / "model_MAPP_GCluster_v4.tar.gz"
        
        if self.compress_directory_tar(mapp_gcluster_v4_dir, mapp_gcluster_v4_output):
            success_count += 1
        
        # 5. Compress MAPPPure model (version_2)
        logger.info("\n[5/10] Compressing MAPPPure model...")
        mapp_pure_dir = self.project_root / "MOFTransformer" / "logs" / "ads_co2_n2_pure_v4_seed42_extranformerv4_from_pmtransformer" / "version_2"
        mapp_pure_output = self.project_root / "MOFTransformer" / "logs" / "ads_co2_n2_pure_v4_seed42_extranformerv4_from_pmtransformer" / "model_MAPPPure.tar.gz"
        
        if self.compress_directory_tar(mapp_pure_dir, mapp_pure_output):
            success_count += 1
        
        # 6. Compress CGCNN GMOF model (version_7)
        logger.info("\n[6/10] Compressing CGCNN GMOF model...")
        cgcnn_gmof_dir = self.project_root / "CGCNN_MT" / "logs" / "SymlogAbsLoadingCO2_SymlogAbsLoadingN2_seed42_cgcnn_langmuir" / "version_7"
        cgcnn_gmof_output = self.project_root / "CGCNN_MT" / "logs" / "SymlogAbsLoadingCO2_SymlogAbsLoadingN2_seed42_cgcnn_langmuir" / "model_CGCNN_GMOF.tar.gz"
        
        if self.compress_directory_tar(cgcnn_gmof_dir, cgcnn_gmof_output):
            success_count += 1
        
        # 7. Compress CGCNN GCluster model (version_8)
        logger.info("\n[7/10] Compressing CGCNN GCluster model...")
        cgcnn_gcluster_dir = self.project_root / "CGCNN_MT" / "logs" / "SymlogAbsLoadingCO2_SymlogAbsLoadingN2_seed42_cgcnn_langmuir" / "version_8"
        cgcnn_gcluster_output = self.project_root / "CGCNN_MT" / "logs" / "SymlogAbsLoadingCO2_SymlogAbsLoadingN2_seed42_cgcnn_langmuir" / "model_CGCNN_GCluster.tar.gz"
        
        if self.compress_directory_tar(cgcnn_gcluster_dir, cgcnn_gcluster_output):
            success_count += 1
        
        # 8. Compress exp_mof_data directory
        logger.info("\n[8/10] Compressing exp_mof_data...")
        exp_mof_dir = self.project_root / "CGCNN_MT" / "data" / "exp_mof_data"
        exp_mof_output = self.project_root / "CGCNN_MT" / "data" / "exp_mof_data.tar.gz"
        
        if self.compress_directory_tar(exp_mof_dir, exp_mof_output):
            success_count += 1
        
        # 9. Compress isotherm extra test data
        logger.info("\n[9/10] Compressing isotherm extra test data...")
        isotherm_file = self.project_root / "CGCNN_MT" / "data" / "ddmof" / "00-isotherm-data-ddmof_extra_test100.tsv"
        isotherm_output = self.project_root / "CGCNN_MT" / "data" / "ddmof" / "isotherm_extra_test.tar.gz"
        
        if self.compress_file_tar(isotherm_file, isotherm_output):
            success_count += 1
        
        # 10. Compress processed data
        logger.info("\n[10/10] Compressing processed data...")
        processed_file = self.project_root / "CGCNN_MT" / "data" / "ddmof" / "id_condition_ads_qst_org_all.csv"
        processed_output = self.project_root / "CGCNN_MT" / "data" / "ddmof" / "processed_data.tar.gz"
        
        if self.compress_file_tar(processed_file, processed_output):
            success_count += 1
        
        logger.info("\n" + "=" * 80)
        logger.info(f"Compression completed: {success_count}/{total_tasks} tasks successful")
        logger.info("=" * 80)
        
        return success_count == total_tasks
    
    def generate_compression_report(self) -> str:
        """Generate a detailed compression report."""
        if not self.compression_info:
            return "No compression operations performed."
        
        report = "\n" + "=" * 80 + "\n"
        report += "COMPRESSION REPORT FOR ZENODO VERSION 2\n"
        report += "=" * 80 + "\n"
        
        total_original_size = 0
        total_compressed_size = 0
        
        for i, info in enumerate(self.compression_info, 1):
            report += f"\n{i}. {Path(info['source']).name}\n"
            report += f"   Output: {info['output']}\n"
            report += f"   Original Size: {info['original_size_mb']:.2f} MB\n"
            report += f"   Compressed Size: {info['compressed_size_mb']:.2f} MB\n"
            report += f"   Compression Ratio: {info['compression_ratio']:.1f}%\n"
            
            if 'file_count' in info:
                report += f"   Files Compressed: {info['file_count']}\n"
            
            total_original_size += info['original_size_mb']
            total_compressed_size += info['compressed_size_mb']
        
        overall_ratio = (1 - total_compressed_size / total_original_size) * 100 if total_original_size > 0 else 0
        
        report += "\n" + "-" * 80 + "\n"
        report += "SUMMARY\n"
        report += "-" * 80 + "\n"
        report += f"Total Original Size: {total_original_size:.2f} MB ({total_original_size/1024:.2f} GB)\n"
        report += f"Total Compressed Size: {total_compressed_size:.2f} MB ({total_compressed_size/1024:.2f} GB)\n"
        report += f"Overall Compression Ratio: {overall_ratio:.1f}%\n"
        report += f"Space Saved: {total_original_size - total_compressed_size:.2f} MB\n"
        report += "=" * 80 + "\n"
        
        return report
    
    def list_output_files(self) -> List[str]:
        """List all output files for upload."""
        output_files = [
            "CGCNN_MT/data/ddmof/mof_cluster_split_val1_test3_seed0_org_csv.zip",
            "CGCNN_MT/data/ddmof/mof_split_val1000_test1000_seed0_org_csv.zip",
            "MOFTransformer/logs/ads_co2_n2_org_v4_seed42_extranformerv4_from_pmtransformer/model_MAPP_GMOF_v4.tar.gz",
            "MOFTransformer/logs/ads_co2_n2_org_v4_seed42_extranformerv4_from_pmtransformer/model_MAPP_GCluster_v4.tar.gz",
            "MOFTransformer/logs/ads_co2_n2_pure_v4_seed42_extranformerv4_from_pmtransformer/model_MAPPPure.tar.gz",
            "CGCNN_MT/logs/SymlogAbsLoadingCO2_SymlogAbsLoadingN2_seed42_cgcnn_langmuir/model_CGCNN_GMOF.tar.gz",
            "CGCNN_MT/logs/SymlogAbsLoadingCO2_SymlogAbsLoadingN2_seed42_cgcnn_langmuir/model_CGCNN_GCluster.tar.gz",
            "CGCNN_MT/data/exp_mof_data.tar.gz",
            "CGCNN_MT/data/ddmof/isotherm_extra_test.tar.gz",
            "CGCNN_MT/data/ddmof/processed_data.tar.gz",
        ]
        return output_files


def main():
    """Main function to run the compression script."""
    try:
        # Get project root directory
        project_root = Path(__file__).parent
        
        logger.info(f"Starting data and model compression for project: {project_root}")
        
        # Create compressor instance
        compressor = DataModelCompressor(str(project_root))
        
        # Compress all targets for Version 2
        success = compressor.compress_all_targets_v2()
        
        # Generate and display report
        report = compressor.generate_compression_report()
        print(report)
        
        # Save report to file
        report_file = project_root / "compression_report_v2.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        
        logger.info(f"Compression report saved to: {report_file}")
        
        # List output files
        print("\nFiles ready for Zenodo upload:")
        for f in compressor.list_output_files():
            full_path = project_root / f
            if full_path.exists():
                size_mb = full_path.stat().st_size / (1024 * 1024)
                print(f"  ✓ {f} ({size_mb:.2f} MB)")
            else:
                print(f"  ✗ {f} (NOT FOUND)")
        
        if success:
            logger.info("\nAll compression tasks completed successfully!")
            sys.exit(0)
        else:
            logger.error("\nSome compression tasks failed. Check the logs above.")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
