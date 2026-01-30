import os
from celery import Celery
from celery.utils.log import get_task_logger

app = Celery('tasks', broker=os.getenv("CELERY_BROKER_URL"))
logger = get_task_logger(__name__)


# @app.task
# def add(x, y):
#     logger.info(f'Adding {x} + {y}')
#     return x + y


@app.task(bind=True)
def analyze_video_task(self, video_path: str, mode: str, sampling_mode: str):
    """Celery task for video analysis"""
    job_id = self.request.id
    output_dir = f"/tmp/frame_analysis/{job_id}"
    
    try:
        engine = FrameAnalysisEngine(job_id, output_dir)
        result = engine.analyze_video(
            video_path,
            AnalysisMode(mode),
            SamplingMode(sampling_mode)
        )
        
        # Save result to JSON
        result_path = os.path.join(output_dir, "result.json")
        with open(result_path, 'w') as f:
            json.dump(asdict(result), f, default=str, indent=2)

        # ========== ADD THIS: Generate PDF ==========
        pdf_path = None
        try:
            from pdfgneration import generate_pdf_report
            pdf_path = generate_pdf_report(result_path)
            logger.info(f"PDF report generated: {pdf_path}")
        except Exception as e:
            logger.error(f"PDF generation failed: {e}")
        # ========== END PDF GENERATION ==========
        
        return {
            'status': 'completed',
            'result_path': result_path,
            'pdf_path': pdf_path,
            'findings_count': len(result.findings)
        }
    
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return {
            'status': 'failed',
            'error': str(e)
        }

@app.task(bind=True)
def analyze_image_sequence_task(self, image_paths: List[str], sequence_type: str, mode: str):
    """Celery task for image sequence analysis"""
    job_id = self.request.id
    output_dir = f"/tmp/frame_analysis/{job_id}"
    
    try:
        engine = FrameAnalysisEngine(job_id, output_dir)
        result = engine.analyze_image_sequence(
            image_paths,
            SequenceType(sequence_type),
            AnalysisMode(mode)
        )
        
        # Save result to JSON
        result_path = os.path.join(output_dir, "result.json")
        with open(result_path, 'w') as f:
            json.dump(asdict(result), f, default=str, indent=2)

         #Generate PDF
        pdf_path = None
        try:
            from pdf_generator import generate_pdf_report
            pdf_path = generate_pdf_report(result_path)
            logger.info(f"PDF report generated: {pdf_path}")
        except Exception as e:
            logger.error(f"PDF generation failed: {e}")
        
        
        return {
            'status': 'completed',
            'result_path': result_path,
            'findings_count': len(result.findings)
        }
    
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return {
            'status': 'failed',
            'error': str(e)
        }
