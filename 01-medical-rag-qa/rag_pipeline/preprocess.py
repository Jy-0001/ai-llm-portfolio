import os
import glob
import logging
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Optional
from pathlib import Path
import pdfplumber


# 工业级PDF批量处理器, 生产一线级别的代码
class PDFBatchProcessor:
    def __init__(self, output_dir: str = "./output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # 配置日志系统
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / "pdf_processing.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    # 查找指定路径下的所有PDF文件
    def find_pdf_files(self, input_path: str) -> List[Path]:
        path = Path(input_path)
        if path.is_file() and path.suffix.lower() == '.pdf':
            return [path]
        elif path.is_dir():
            # 递归查找所有PDF文件
            pdf_files = list(path.glob("**/*.pdf"))
            self.logger.info(f"在 {input_path} 中找到 {len(pdf_files)} 个PDF文件")
            return pdf_files
        else:
            raise ValueError(f"路径不存在,或不是PDF文件: {input_path}")

    # 提取单个PDF文件的内容
    def extract_pdf_content(self,
                            pdf_path: Path,
                            extract_text: bool = True,
                            extract_tables: bool = True,
                            table_settings: Optional[dict] = None) -> Dict:
        """
        Args:
            pdf_path: PDF文件路径
            extract_text: 是否提取文本
            extract_tables: 是否提取表格
            table_settings: 表格提取配置
        """
        result = {
            "file_name": pdf_path.name,
            "file_path": str(pdf_path),
            "metadata": {},
            "pages": [],
            "error": None
        }
        try:
            with pdfplumber.open(pdf_path) as pdf:
                # 提取元数据
                result["metadata"] = pdf.metadata

                for page_num, page in enumerate(pdf.pages, 1):
                    page_result = {"page_number": page_num, "text": "", "tables": []}

                    # 提取文本
                    if extract_text:
                        try:
                            # 布局模式根据需求调整
                            text = page.extract_text(layout=False)
                            page_result["text"] = text if text else ""
                        except Exception as e:
                            self.logger.warning(f"页面 {page_num} 文本提取失败: {str(e)}")
                            pass

                    # 提取表格
                    if extract_tables:
                        try:
                            tables = page.extract_tables(table_settings or {})
                            if tables:
                                page_result["tables"] = tables
                        except Exception as e:
                            self.logger.warning(f"页面 {page_num} 表格提取失败: {str(e)}")
                            pass

                    # 添加当前页面page的提取结果
                    result["pages"].append(page_result)

                # 单一PDF文档提取完毕后, 写日志处理
                self.logger.info(f"成功处理: {pdf_path.name} - {len(pdf.pages)} 页")

        # 单一PDF文档提取失败后, 写日志处理
        except Exception as e:
            # 明确记录一下哪篇PDF文档处理失败, 并记录失败原因, 便于后续回溯与 "bad case分析"
            error_msg = f"处理文件失败 {pdf_path}: {str(e)}"
            result["error"] = error_msg
            self.logger.error(error_msg)

        return result

    # 批量处理PDF文件
    def process_batch(self, pdf_files: List[Path],
                      save_format: str = "excel",
                      **extract_kwargs) -> pd.DataFrame:
        """
        Args:
            pdf_files: PDF文件列表
            save_format: 保存格式 (excel, csv, parquet)
            **extract_kwargs: 提取参数
        """
        all_results = []

        for i, pdf_file in tqdm(enumerate(pdf_files, 1)):
            self.logger.info(f"处理进度: {i}/{len(pdf_files)} - {pdf_file.name}")

            result = self.extract_pdf_content(pdf_file, **extract_kwargs)
            all_results.append(result)

            # 实时保存进度 (针对大批量处理)
            if i % 10 == 0:
                self._save_intermediate_results(all_results, f"batch_{i}")

        # 保存最终结果
        return self._save_results(all_results, save_format)

    # 保存处理结果
    def _save_results(self, results: List[Dict], format: str) -> pd.DataFrame:
        # 扁平化结果, 以便保存
        flat_data = []

        for result in results:
            if result["error"]:
                flat_data.append(
                    {
                        "file_name": result["file_name"],
                        "status": "Error",
                        "error_message": result["error"],
                        "page_count": 0,
                        "text_length": 0,
                        "table_count": 0
                    }
                )
                continue

            total_text = ""
            total_tables = 0
            for page in result["pages"]:
                total_text += page["text"]
                total_tables += len(page["tables"])

            flat_data.append({
                "file_name": result["file_name"],
                "status": "Success",
                "error_message": "",
                "page_count": len(result["pages"]),
                "text_length": len(total_text),
                "table_count": total_tables,
                "author": result["metadata"].get("Author", ""),
                "creation_date": result["metadata"].get("CreationDate", "")
            })

        # for循环处理完毕后, 所有数据封装成 Pandas 的 DataFrame 格式
        df = pd.DataFrame(flat_data)

        # 根据格式保存
        if format.lower() == "excel":
            df.to_excel(self.output_dir / "pdf_extraction_summary.xlsx", index=False)

            # 同时保存详细文本内容
            detailed_results = []
            for result in results:
                if not result["error"]:
                    for page in result["pages"]:
                        if page["text"]:
                            detailed_results.append({
                                "file_name": result["file_name"],
                                "page_number": page["page_number"],
                                "text_content": page["text"]
                            })

            if detailed_results:
                pd.DataFrame(detailed_results).to_excel(
                    self.output_dir / "pdf_detailed_text.xlsx", index=False
                )

        elif format.lower() == "csv":
            df.to_csv(self.output_dir / "pdf_extraction_summary.csv", index=False)

        self.logger.info(f"结果已保存到 {self.output_dir}")
        return df

    # 保存中间结果 (工业界一线生产环境, 异常因素很多, 防止处理中断丢失数据)
    def _save_intermediate_results(self, results: List[Dict], batch_name: str):
        try:
            temp_df = pd.DataFrame([{
                "file_name": r["file_name"],
                "status": "Error" if r["error"] else "Success",
                "pages_processed": len(r["pages"])
            } for r in results])

            temp_df.to_csv(self.output_dir / f"progress_{batch_name}.csv", index=False)
        except Exception as e:
            self.logger.warning(f"保存中间结果失败: {str(e)}")


# 高级表格提取配置
ADVANCED_TABLE_SETTINGS = {
    "vertical_strategy": "lines",
    "horizontal_strategy": "lines",
    "snap_tolerance": 4,
    "join_tolerance": 10,
    "edge_min_length": 3,
    "min_words_vertical": 2,
    "min_words_horizontal": 1
}


def main():
    # 实例化PDF处理器对象
    processor = PDFBatchProcessor(output_dir="./pdf_output")

    try:
        # 查找PDF文件
        pdf_files = processor.find_pdf_files("./pdf_documents")

        if not pdf_files:
            processor.logger.warning("未找到PDF文件")
            return

        # 批量处理
        results_df = processor.process_batch(
            pdf_files,
            save_format="excel",
            extract_text=True,
            extract_tables=True,
            table_settings=ADVANCED_TABLE_SETTINGS
        )

        # 打印摘要统计
        success_count = len(results_df[results_df["status"] == "Success"])
        processor.logger.info(f"处理完成: {success_count}/{len(pdf_files)} 个文件成功")

        if success_count > 0:
            avg_text_length = results_df[results_df["status"] == "Success"]["text_length"].mean()
            avg_tables = results_df[results_df["status"] == "Success"]["table_count"].mean()
            processor.logger.info(f"平均每文件: {avg_text_length:.0f} 字符, {avg_tables:.1f} 个表格")

    # 处理过程中发生错误, 记录日志
    except Exception as e:
        processor.logger.error(f"处理过程发生错误: {str(e)}")


if __name__ == "__main__":
    main()