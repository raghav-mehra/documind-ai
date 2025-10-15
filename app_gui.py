# app_gui.py
import customtkinter as ctk
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import time
import threading
import queue
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
import json
from datetime import datetime

# Import your existing classes
from Engine.ocr_processor import OCRProcessor
from Engine.smart_batch_analyzer import SmartBatchAnalyzer, BatchAnalysisResult
from Engine.ai_analyser import AnalysisType


class ThreadSafeGUI:
    """Utility class for thread-safe GUI operations"""
    
    def __init__(self, root):
        self.root = root
        self.task_queue = queue.Queue()
        
    def schedule_task(self, task, *args):
        """Schedule a task to run in the main thread"""
        self.task_queue.put((task, args))
        
    def process_tasks(self):
        """Process all pending GUI tasks (call this periodically)"""
        try:
            while True:
                task, args = self.task_queue.get_nowait()
                try:
                    task(*args)
                except Exception as e:
                    print(f"Error executing GUI task: {e}")
                self.task_queue.task_done()
        except queue.Empty:
            pass
        
        # Schedule next check
        self.root.after(50, self.process_tasks)


class SplashScreen:
    """Professional splash screen for app initialization"""
    
    def __init__(self, gui_manager):
        self.gui_manager = gui_manager
        self.splash = None
        self.progress_bar = None
        self.status_label = None
        
    def show(self):
        """Show the splash screen"""
        self.splash = ctk.CTkToplevel()
        self.splash.title("Initializing...")
        self.splash.geometry("500x400")
        self.splash.resizable(False, False)
        
        # Center and make modal
        self.splash.transient()
        self.splash.grab_set()
        self.splash.overrideredirect(True)
        self.splash.configure(fg_color="#2b2b2b")
        
        self._setup_ui()
        self._center_on_screen()
        
    def _setup_ui(self):
        """Setup splash screen UI"""
        main_frame = ctk.CTkFrame(self.splash, fg_color="transparent")
        main_frame.pack(expand=True, fill="both", padx=40, pady=40)
        
        # App logo/icon
        logo_label = ctk.CTkLabel(
            main_frame,
            text="📄",
            font=ctk.CTkFont(size=80, weight="bold"),
            text_color="#4CC9F0"
        )
        logo_label.pack(pady=(20, 10))
        
        # App name
        app_name_label = ctk.CTkLabel(
            main_frame,
            text="DocuMind AI",
            font=ctk.CTkFont(size=32, weight="bold", family="Segoe UI"),
            text_color="white"
        )
        app_name_label.pack(pady=(0, 5))
        
        # Tagline
        tagline_label = ctk.CTkLabel(
            main_frame,
            text="Intelligent Document Processing & Analysis",
            font=ctk.CTkFont(size=14, family="Segoe UI"),
            text_color="#B0B0B0"
        )
        tagline_label.pack(pady=(0, 30))
        
        # Progress bar
        self.progress_bar = ctk.CTkProgressBar(
            main_frame,
            width=300,
            height=4,
            progress_color="#4CC9F0"
        )
        self.progress_bar.pack(pady=20)
        self.progress_bar.set(0)
        
        # Status label
        self.status_label = ctk.CTkLabel(
            main_frame,
            text="Initializing application...",
            font=ctk.CTkFont(size=12, family="Segoe UI"),
            text_color="#808080"
        )
        self.status_label.pack(pady=10)
        
        # Version info
        version_label = ctk.CTkLabel(
            main_frame,
            text="Version 1.0.0",
            font=ctk.CTkFont(size=10, family="Segoe UI"),
            text_color="#606060"
        )
        version_label.pack(side="bottom", pady=10)
        
    def _center_on_screen(self):
        """Center the splash screen on the display"""
        self.splash.update_idletasks()
        width = self.splash.winfo_width()
        height = self.splash.winfo_height()
        x = (self.splash.winfo_screenwidth() // 2) - (width // 2)
        y = (self.splash.winfo_screenheight() // 2) - (height // 2)
        self.splash.geometry(f"{width}x{height}+{x}+{y}")
        
    def update_status(self, status: str, progress: float):
        """Update status and progress - thread-safe"""
        self.gui_manager.schedule_task(self._update_ui, status, progress)
        
    def _update_ui(self, status: str, progress: float):
        """Update UI elements (called in main thread)"""
        if self.status_label and self.progress_bar:
            self.status_label.configure(text=status)
            self.progress_bar.set(progress)
            self.splash.update()
        
    def close(self):
        """Close the splash screen"""
        if self.splash:
            self.splash.destroy()
            self.splash = None


class DocumentCard:
    """Individual document card widget"""
    
    def __init__(self, parent, doc_data, selection_callback):
        self.parent = parent
        self.doc_data = doc_data
        self.selection_callback = selection_callback
        self.is_selected = False
        
        self.card = None
        self.selection_indicator = None
        self.status_label = None
        
        self.create_widgets()
        
    def create_widgets(self):
        """Create the card widgets"""
        self.card = ctk.CTkFrame(
            self.parent,
            border_width=1,
            border_color="#3b3b3b",
            corner_radius=10,
            width=200,
            height=280
        )
        self.card.pack_propagate(False)
        
        # Thumbnail/icon
        thumbnail_label = ctk.CTkLabel(
            self.card,
            text=self.doc_data.get('thumbnail', '📄'),
            font=ctk.CTkFont(size=48),
            text_color="#4CC9F0"
        )
        thumbnail_label.pack(pady=(20, 10))
        
        # File name (truncated if too long)
        file_name = self.doc_data['file_name']
        if len(file_name) > 20:
            file_name = file_name[:17] + "..."
            
        name_label = ctk.CTkLabel(
            self.card,
            text=file_name,
            font=ctk.CTkFont(size=12, weight="bold"),
            wraplength=180
        )
        name_label.pack(pady=(0, 5))
        
        # Status
        self.status_label = ctk.CTkLabel(
            self.card,
            text=self.doc_data['status'],
            font=ctk.CTkFont(size=11),
            text_color=self.get_status_color(self.doc_data['status'])
        )
        self.status_label.pack(pady=(0, 5))
        
        # Selection indicator
        self.selection_indicator = ctk.CTkFrame(
            self.card,
            width=10,
            height=10,
            corner_radius=5,
            fg_color="transparent",
            border_width=1,
            border_color="#4CC9F0"
        )
        self.selection_indicator.place(relx=0.9, rely=0.1, anchor="center")
        
        # Bind click event
        self.card.bind("<Button-1>", self.on_click)
        for child in self.card.winfo_children():
            child.bind("<Button-1>", self.on_click)
            
        self.update_appearance()
        
    def get_status_color(self, status):
        """Get color for status text"""
        colors = {
            'Pending': '#B0B0B0',
            'Processing...': '#FFA500',
            'Page 1 processed': '#4CC9F0', 
            'Analyzed': '#4CAF50',
            'Complete': '#4CAF50',
            'Error': '#F44336'
        }
        return colors.get(status, '#B0B0B0')
        
    def on_click(self, event):
        """Handle card click"""
        self.is_selected = not self.is_selected
        self.update_appearance()
        self.selection_callback(self.doc_data['file_path'], self.is_selected)
        return "break"  # Prevent event propagation
        
    def update_appearance(self):
        """Update card appearance based on selection state"""
        if self.selection_indicator and self.card:
            self.selection_indicator.configure(
                fg_color="#4CC9F0" if self.is_selected else "transparent"
            )
            self.card.configure(
                border_color="#4CC9F0" if self.is_selected else "#3b3b3b"
            )
            
    def update_status(self, status: str):
        """Update document status"""
        self.doc_data['status'] = status
        if self.status_label:
            self.status_label.configure(
                text=status,
                text_color=self.get_status_color(status)
            )
            
    def destroy(self):
        """Destroy the card"""
        if self.card:
            self.card.destroy()


class DocumentTab:
    """Tab for managing documents"""
    
    def __init__(self, parent, tab_control, tab_name):
        self.parent = parent
        self.tab_control = tab_control
        self.tab_name = tab_name
        
        # Create tab
        self.tab_control.add(tab_name)
        self.tab_frame = self.tab_control.tab(tab_name)
        
        self.documents: Dict[str, dict] = {}  # file_path -> document_data
        self.selected_documents: Set[str] = set()
        self.document_cards: Dict[str, DocumentCard] = {}
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup tab UI"""
        # Main content area
        self.content_frame = ctk.CTkFrame(self.tab_frame)
        self.content_frame.pack(expand=True, fill="both", padx=10, pady=10)
        
        # Setup views
        self.setup_drag_drop_area()
        self.setup_documents_grid()
        
        # Show appropriate view
        self.update_view()
        
    def setup_drag_drop_area(self):
        """Setup drag drop area for empty tab"""
        self.drag_drop_frame = ctk.CTkFrame(
            self.content_frame,
            border_width=2,
            border_color="#4CC9F0",
            corner_radius=15,
            fg_color="#1E1E1E"
        )
        
        inner_frame = ctk.CTkFrame(self.drag_drop_frame, fg_color="transparent")
        inner_frame.pack(expand=True, fill="both", padx=40, pady=50)
        
        # Icon
        icon_label = ctk.CTkLabel(
            inner_frame,
            text="📄",
            font=ctk.CTkFont(size=64),
            text_color="#4CC9F0"
        )
        icon_label.pack(pady=(0, 20))
        
        # Title
        title_label = ctk.CTkLabel(
            inner_frame,
            text="Add Documents for Analysis",
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color="white"
        )
        title_label.pack(pady=(0, 10))
        
        # Description
        desc_label = ctk.CTkLabel(
            inner_frame,
            text="Upload PDFs or images for intelligent document processing\nand AI-powered analysis",
            font=ctk.CTkFont(size=14),
            text_color="#B0B0B0",
            justify="center"
        )
        desc_label.pack(pady=(0, 30))
        
        # Upload button
        upload_btn = ctk.CTkButton(
            inner_frame,
            text="📁 Upload Documents",
            command=self.browse_files,
            fg_color="#4CC9F0",
            hover_color="#3AA8D9",
            height=45,
            width=220,
            font=ctk.CTkFont(size=15, weight="bold"),
            corner_radius=8
        )
        upload_btn.pack(pady=10)
        
        # File info
        info_label = ctk.CTkLabel(
            inner_frame,
            text="Supported formats: PDF, JPG, PNG, TIFF • Max 50MB per file",
            font=ctk.CTkFont(size=12),
            text_color="#808080"
        )
        info_label.pack(pady=(20, 0))
        
    def setup_documents_grid(self):
        """Setup grid view for documents"""
        self.grid_container = ctk.CTkFrame(self.content_frame, fg_color="transparent")
        
        # Scrollable frame
        self.scrollable_frame = ctk.CTkScrollableFrame(
            self.grid_container,
            fg_color="transparent"
        )
        self.scrollable_frame.pack(expand=True, fill="both", padx=10, pady=10)
        
        # Grid frame
        self.grid_frame = ctk.CTkFrame(self.scrollable_frame, fg_color="transparent")
        self.grid_frame.pack(expand=True, fill="both")
        
    def update_view(self):
        """Update the view based on document count"""
        for widget in self.content_frame.winfo_children():
            widget.pack_forget()
            
        if not self.documents:
            self.drag_drop_frame.pack(expand=True, fill="both", padx=20, pady=20)
        else:
            self.grid_container.pack(expand=True, fill="both", padx=10, pady=10)
            self.update_grid_view()
            
    def update_grid_view(self):
        """Update the grid view with current documents"""
        # Clear existing cards
        for card in self.document_cards.values():
            card.destroy()
        self.document_cards.clear()
        
        # Clear grid frame
        for widget in self.grid_frame.winfo_children():
            widget.destroy()
            
        if not self.documents:
            self.update_view()
            return
            
        # Create grid layout (3 columns)
        row, col = 0, 0
        max_cols = 3
        
        for file_path, doc_data in self.documents.items():
            # Create document card
            card = DocumentCard(
                self.grid_frame, 
                doc_data, 
                self.on_document_selection
            )
            card.card.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")
            
            # Configure grid weights
            self.grid_frame.grid_rowconfigure(row, weight=0)
            self.grid_frame.grid_columnconfigure(col, weight=1)
            
            self.document_cards[file_path] = card
            
            # Update selection state
            if file_path in self.selected_documents:
                card.is_selected = True
                card.update_appearance()
            
            # Update grid position
            col += 1
            if col >= max_cols:
                col = 0
                row += 1
                
    def on_document_selection(self, file_path: str, selected: bool):
        """Handle document selection change"""
        if selected:
            self.selected_documents.add(file_path)
        else:
            self.selected_documents.discard(file_path)
            
        self.parent.update_button_states()
        
    def browse_files(self):
        """Browse and select files"""
        files = filedialog.askopenfilenames(
            title="Select Documents",
            filetypes=[
                ("All Supported", "*.pdf *.jpg *.jpeg *.png *.tiff *.tif"),
                ("PDF Files", "*.pdf"),
                ("Image Files", "*.jpg *.jpeg *.png *.tiff *.tif"),
                ("All Files", "*.*")
            ]
        )
        if files:
            self.add_documents(files)
            
    def add_documents(self, file_paths):
        """Add documents to the tab"""
        for file_path in file_paths:
            if file_path not in self.documents:
                document_data = {
                    'file_path': file_path,
                    'file_name': Path(file_path).name,
                    'status': 'Pending',
                    'thumbnail': '📄',
                    'analysis_result': None
                }
                self.documents[file_path] = document_data
                
        self.update_view()
        self.parent.update_dashboard_stats()
        self.parent.update_button_states()
        
    def select_all_documents(self):
        """Select all documents"""
        self.selected_documents = set(self.documents.keys())
        for file_path, card in self.document_cards.items():
            card.is_selected = True
            card.update_appearance()
        self.parent.update_button_states()
        
    def deselect_all_documents(self):
        """Deselect all documents"""
        self.selected_documents.clear()
        for card in self.document_cards.values():
            card.is_selected = False
            card.update_appearance()
        self.parent.update_button_states()
        
    def get_selected_documents(self):
        """Get list of selected document paths"""
        return list(self.selected_documents)
        
    def update_document_status(self, file_path: str, status: str):
        """Update document status"""
        if file_path in self.documents:
            self.documents[file_path]['status'] = status
            if file_path in self.document_cards:
                self.document_cards[file_path].update_status(status)
                
    def set_analysis_result(self, file_path: str, result: dict):
        """Set analysis result for document"""
        if file_path in self.documents:
            self.documents[file_path]['analysis_result'] = result
            
    def rename_tab(self, new_name: str):
        """Rename this tab"""
        old_name = self.tab_name
        self.tab_name = new_name
        self.parent.rename_document_tab(old_name, new_name)


class DetailAnalysisTab:
    """Tab for detailed document analysis with chat interface"""
    
    def __init__(self, parent, tab_control, file_path: str, analysis_data: Dict):
        self.parent = parent
        self.tab_control = tab_control
        self.file_path = file_path
        self.analysis_data = analysis_data
        self.file_name = Path(file_path).name
        self.tab_name = f"Detail: {self.file_name}"
        
        # Chat history
        self.chat_history: List[Dict] = []
        
        # Create tab
        self.tab_control.add(self.tab_name)
        self.tab_frame = self.tab_control.tab(self.tab_name)
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup detailed analysis UI"""
        # Main container with grid layout
        self.tab_frame.grid_rowconfigure(0, weight=0)  # Header
        self.tab_frame.grid_rowconfigure(1, weight=1)  # Content area
        self.tab_frame.grid_rowconfigure(2, weight=0)  # Input area
        self.tab_frame.grid_columnconfigure(0, weight=1)
        self.tab_frame.grid_columnconfigure(1, weight=0)  # View toggle
        
        # Header
        self.setup_header()
        
        # Content area (data view + chat)
        self.setup_content_area()
        
        # Input area
        self.setup_input_area()
        
    def setup_header(self):
        """Setup header with view toggle"""
        header_frame = ctk.CTkFrame(self.tab_frame, fg_color="transparent")
        header_frame.grid(row=0, column=0, columnspan=2, sticky="ew", padx=20, pady=10)
        
        # Title
        title_label = ctk.CTkLabel(
            header_frame,
            text=f"Detailed Analysis: {self.file_name}",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        title_label.pack(side="left")
        
        # View toggle
        view_toggle_frame = ctk.CTkFrame(header_frame, fg_color="transparent")
        view_toggle_frame.pack(side="right")
        
        ctk.CTkLabel(
            view_toggle_frame,
            text="View:",
            font=ctk.CTkFont(size=12)
        ).pack(side="left", padx=(0, 10))
        
        self.view_var = ctk.StringVar(value="hierarchical")
        
        hierarchical_btn = ctk.CTkRadioButton(
            view_toggle_frame,
            text="Hierarchical",
            variable=self.view_var,
            value="hierarchical",
            command=self.toggle_view
        )
        hierarchical_btn.pack(side="left", padx=(0, 10))
        
        pointwise_btn = ctk.CTkRadioButton(
            view_toggle_frame,
            text="Point-wise",
            variable=self.view_var,
            value="pointwise",
            command=self.toggle_view
        )
        pointwise_btn.pack(side="left")
        
    def setup_content_area(self):
        """Setup content area with data view and chat"""
        content_frame = ctk.CTkFrame(self.tab_frame)
        content_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=20, pady=(0, 10))
        content_frame.grid_rowconfigure(0, weight=1)
        content_frame.grid_columnconfigure(0, weight=1)
        content_frame.grid_columnconfigure(1, weight=0)  # Separator
        
        # Data view (left side)
        self.setup_data_view(content_frame)
        
        # Separator
        separator = ctk.CTkFrame(content_frame, width=2, fg_color="#3b3b3b")
        separator.grid(row=0, column=1, sticky="ns", padx=10)
        
        # Chat interface (right side)
        self.setup_chat_interface(content_frame)
        
    def setup_data_view(self, parent):
        """Setup data view area"""
        data_frame = ctk.CTkFrame(parent, fg_color="transparent")
        data_frame.grid(row=0, column=0, sticky="nsew")
        data_frame.grid_rowconfigure(1, weight=1)
        data_frame.grid_columnconfigure(0, weight=1)
        
        # Data view header
        data_header = ctk.CTkFrame(data_frame, fg_color="transparent")
        data_header.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        
        ctk.CTkLabel(
            data_header,
            text="Document Analysis Data",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(side="left")
        
        # Data display area
        self.data_display_frame = ctk.CTkFrame(data_frame)
        self.data_display_frame.grid(row=1, column=0, sticky="nsew")
        
        # Initialize with hierarchical view
        self.show_hierarchical_view()
        
    def show_hierarchical_view(self):
        """Show hierarchical tree view of analysis data"""
        # Clear existing widgets
        for widget in self.data_display_frame.winfo_children():
            widget.destroy()
            
        # Create scrollable frame
        scroll_frame = ctk.CTkScrollableFrame(self.data_display_frame)
        scroll_frame.pack(expand=True, fill="both", padx=10, pady=10)
        
        # Create tree structure
        self.create_tree_structure(scroll_frame, self.analysis_data, "")
        
    def create_tree_structure(self, parent, data: Dict, prefix: str, level: int = 0):
        """Recursively create tree structure"""
        for key, value in data.items():
            if isinstance(value, dict):
                # Create expandable section
                section_frame = ctk.CTkFrame(parent, fg_color="transparent")
                section_frame.pack(fill="x", pady=2)
                
                # Expand button and label
                expand_btn = ctk.CTkButton(
                    section_frame,
                    text="▶",
                    width=25,
                    height=25,
                    fg_color="transparent",
                    hover_color="#3b3b3b",
                    command=lambda f=section_frame, k=key, v=value: self.toggle_section(f, k, v)
                )
                expand_btn.pack(side="left")
                
                ctk.CTkLabel(
                    section_frame,
                    text=key,
                    font=ctk.CTkFont(size=12, weight="bold"),
                    anchor="w"
                ).pack(side="left", fill="x", expand=True, padx=(5, 0))
                
                # Content frame (initially hidden)
                content_frame = ctk.CTkFrame(parent, fg_color="transparent")
                
                # Store references for toggling
                section_frame.expand_btn = expand_btn
                section_frame.content_frame = content_frame
                section_frame.is_expanded = False
                
            else:
                # Create key-value pair
                item_frame = ctk.CTkFrame(parent, fg_color="transparent")
                item_frame.pack(fill="x", pady=1)
                
                ctk.CTkLabel(
                    item_frame,
                    text="  " * level + "•",
                    font=ctk.CTkFont(size=10),
                    width=20
                ).pack(side="left")
                
                ctk.CTkLabel(
                    item_frame,
                    text=f"{key}:",
                    font=ctk.CTkFont(size=11, weight="bold"),
                    width=150,
                    anchor="w"
                ).pack(side="left")
                
                value_text = str(value)
                if len(value_text) > 100:
                    value_text = value_text[:100] + "..."
                    
                ctk.CTkLabel(
                    item_frame,
                    text=value_text,
                    font=ctk.CTkFont(size=11),
                    anchor="w",
                    justify="left"
                ).pack(side="left", fill="x", expand=True)
                
    def toggle_section(self, section_frame, key: str, value: Dict):
        """Toggle section expansion"""
        if section_frame.is_expanded:
            section_frame.content_frame.pack_forget()
            section_frame.expand_btn.configure(text="▶")
            section_frame.is_expanded = False
        else:
            section_frame.content_frame.pack(fill="x", before=section_frame)
            # Populate content
            for widget in section_frame.content_frame.winfo_children():
                widget.destroy()
            self.create_tree_structure(section_frame.content_frame, value, "", 1)
            section_frame.expand_btn.configure(text="▼")
            section_frame.is_expanded = True
            
    def show_pointwise_view(self):
        """Show point-wise key-value view"""
        # Clear existing widgets
        for widget in self.data_display_frame.winfo_children():
            widget.destroy()
            
        # Create scrollable frame
        scroll_frame = ctk.CTkScrollableFrame(self.data_display_frame)
        scroll_frame.pack(expand=True, fill="both", padx=10, pady=10)
        
        # Flatten data and display as key-value pairs
        self.display_flattened_data(scroll_frame, self.analysis_data, "")
        
    def display_flattened_data(self, parent, data: Dict, prefix: str):
        """Display data as flattened key-value pairs"""
        for key, value in data.items():
            if isinstance(value, dict):
                self.display_flattened_data(parent, value, f"{prefix}{key}.")
            else:
                item_frame = ctk.CTkFrame(parent, fg_color="transparent")
                item_frame.pack(fill="x", pady=2)
                
                full_key = f"{prefix}{key}"
                ctk.CTkLabel(
                    item_frame,
                    text=full_key,
                    font=ctk.CTkFont(size=11, weight="bold"),
                    width=200,
                    anchor="w"
                ).pack(side="left")
                
                value_text = ctk.CTkTextbox(item_frame, height=min(10, max(10, len(str(value)) // 50 + 10)))
                value_text.pack(side="left", fill="x", expand=True, padx=(10, 0))
                value_text.insert("1.0", str(value))
                value_text.configure(state="disabled")
                
    def toggle_view(self):
        """Toggle between hierarchical and point-wise views"""
        if self.view_var.get() == "hierarchical":
            self.show_hierarchical_view()
        else:
            self.show_pointwise_view()
            
    def setup_chat_interface(self, parent):
        """Setup chat interface"""
        chat_frame = ctk.CTkFrame(parent, fg_color="transparent")
        chat_frame.grid(row=0, column=2, sticky="nsew")
        chat_frame.grid_rowconfigure(1, weight=1)
        chat_frame.grid_columnconfigure(0, weight=1)
        
        # Chat header
        chat_header = ctk.CTkFrame(chat_frame, fg_color="transparent")
        chat_header.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        
        ctk.CTkLabel(
            chat_header,
            text="Chat with Document",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(side="left")
        
        # Chat history area
        self.chat_history_frame = ctk.CTkScrollableFrame(
            chat_frame,
            fg_color="#1a1a1a",
            corner_radius=8
        )
        self.chat_history_frame.grid(row=1, column=0, sticky="nsew", pady=(0, 10))
        
        # Add welcome message
        self.add_chat_message("assistant", "Hello! I can answer questions about this document. What would you like to know?")
        
    def setup_input_area(self):
        """Setup chat input area"""
        input_frame = ctk.CTkFrame(self.tab_frame, fg_color="transparent")
        input_frame.grid(row=2, column=0, columnspan=2, sticky="ew", padx=20, pady=10)
        
        # Input field
        self.chat_input = ctk.CTkTextbox(
            input_frame,
            height=60,
            wrap="word",
            font=ctk.CTkFont(size=12)
        )
        self.chat_input.pack(side="left", fill="x", expand=True, padx=(0, 10))
        self.chat_input.bind("<Return>", self.on_send_message)
        self.chat_input.bind("<Control-Return>", lambda e: "break")  # Allow newline with Ctrl+Enter
        
        # Send button
        send_btn = ctk.CTkButton(
            input_frame,
            text="Send",
            command=self.send_message,
            width=80,
            height=60,
            fg_color="#4CC9F0",
            hover_color="#3AA8D9",
            font=ctk.CTkFont(size=12, weight="bold")
        )
        send_btn.pack(side="right")
        
    def add_chat_message(self, role: str, content: str):
        """Add a message to chat history"""
        message_frame = ctk.CTkFrame(
            self.chat_history_frame,
            fg_color="#2b2b2b" if role == "user" else "#1e1e1e",
            corner_radius=10
        )
        message_frame.pack(fill="x", pady=5, padx=5)
        
        # Role indicator
        role_label = ctk.CTkLabel(
            message_frame,
            text="You:" if role == "user" else "AI:",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#4CC9F0" if role == "user" else "#4CAF50",
            width=40,
            anchor="w"
        )
        role_label.pack(anchor="w", padx=10, pady=(10, 0))
        
        # Message content
        content_text = ctk.CTkTextbox(
            message_frame,
            height=min(10, max(3, len(content) // 50 + 1)),
            wrap="word",
            fg_color="transparent",
            border_width=0,
            font=ctk.CTkFont(size=12)
        )
        content_text.pack(fill="x", padx=10, pady=(0, 10))
        content_text.insert("1.0", content)
        content_text.configure(state="disabled")
        
        # Store in history
        self.chat_history.append({"role": role, "content": content})
        
        # Auto-scroll to bottom
        self.chat_history_frame._parent_canvas.yview_moveto(1.0)
        
    def send_message(self):
        """Send chat message"""
        message = self.chat_input.get("1.0", "end-1c").strip()
        if not message:
            return
            
        # Clear input
        self.chat_input.delete("1.0", "end")
        
        # Add user message
        self.add_chat_message("user", message)
        
        # Process AI response in background
        threading.Thread(target=self.get_ai_response, args=(message,), daemon=True).start()
        
    def on_send_message(self, event):
        """Handle Enter key for sending message"""
        if event.state & 0x4:  # Ctrl key pressed
            return  # Allow newline
        else:
            self.send_message()
            return "break"  # Prevent default behavior
            
    def get_ai_response(self, user_message: str):
        """Get AI response for the chat message"""
        try:
            # Simulate AI processing (replace with actual AI call)
            time.sleep(1)
            
            # Mock response based on document content
            response = self.generate_mock_response(user_message)
            
            # Schedule UI update in main thread
            self.parent.gui_manager.schedule_task(self.add_chat_message, "assistant", response)
            
        except Exception as e:
            error_msg = f"Sorry, I encountered an error: {str(e)}"
            self.parent.gui_manager.schedule_task(self.add_chat_message, "assistant", error_msg)
            
    def generate_mock_response(self, user_message: str) -> str:
        """Generate mock AI response based on document content"""
        # This is a mock implementation - replace with actual AI integration
        lower_message = user_message.lower()
        
        if "summary" in lower_message or "summarize" in lower_message:
            return self.analysis_data.get('summary', 'No summary available in the document analysis.')
        elif "category" in lower_message or "type" in lower_message:
            category = self.analysis_data.get('category', 'Unknown')
            return f"This document is categorized as: {category}"
        elif "key point" in lower_message or "important" in lower_message:
            key_points = self.analysis_data.get('key_points', [])
            if key_points:
                points = "\n".join([f"• {point}" for point in key_points[:3]])
                return f"Key points from the document:\n{points}"
            else:
                return "No key points extracted from this document."
        else:
            return f"I've analyzed your question about the document. Based on the content, I can provide more specific information if you ask about the summary, category, or key points."


class AnalysisResultsTab:
    """Tab for viewing analysis results"""
    
    def __init__(self, parent, tab_control):
        self.parent = parent
        self.tab_control = tab_control
        
        # Create tab
        self.tab_control.add("Analysis Results")
        self.tab_frame = self.tab_control.tab("Analysis Results")
        self.setup_ui()
        
    def setup_ui(self):
        """Setup analysis results UI"""
        # Header
        header_frame = ctk.CTkFrame(self.tab_frame, fg_color="transparent")
        header_frame.pack(fill="x", padx=20, pady=20)
        
        ctk.CTkLabel(
            header_frame,
            text="Document Analysis Results",
            font=ctk.CTkFont(size=20, weight="bold")
        ).pack(side="left")
        
        # Clear button
        clear_btn = ctk.CTkButton(
            header_frame,
            text="Clear All",
            command=self.clear_results,
            fg_color="#C62828",
            hover_color="#B71C1C",
            width=100
        )
        clear_btn.pack(side="right")

        # Results container
        self.results_frame = ctk.CTkScrollableFrame(
            self.tab_frame,
            fg_color="transparent"
        )
        self.results_frame.pack(expand=True, fill="both", padx=20, pady=(0, 20))
        
    def add_result(self, file_path: str, result_data: Dict):
        """Add analysis result to the tab"""
        result_card = ctk.CTkFrame(
            self.results_frame,
            border_width=1,
            border_color="#3b3b3b",
            corner_radius=8
        )
        result_card.pack(fill="x", pady=(0, 10))
        
        # Header
        header_frame = ctk.CTkFrame(result_card, fg_color="transparent")
        header_frame.pack(fill="x", padx=15, pady=10)
        
        # File info
        file_info_frame = ctk.CTkFrame(header_frame, fg_color="transparent")
        file_info_frame.pack(side="left", fill="x", expand=True)
        
        ctk.CTkLabel(
            file_info_frame,
            text=Path(file_path).name,
            font=ctk.CTkFont(size=14, weight="bold"),
            anchor="w"
        ).pack(fill="x")
        
        ctk.CTkLabel(
            file_info_frame,
            text=f"Category: {result_data.get('category', 'Unknown')}",
            font=ctk.CTkFont(size=12),
            text_color="#B0B0B0",
            anchor="w"
        ).pack(fill="x")
        
        # Confidence badge
        confidence = result_data.get('confidence', 0)
        confidence_color = "#4CAF50" if confidence > 0.8 else "#FF9800" if confidence > 0.6 else "#F44336"
        
        confidence_frame = ctk.CTkFrame(header_frame, fg_color="transparent")
        confidence_frame.pack(side="right")
        
        ctk.CTkLabel(
            confidence_frame,
            text=f"Confidence: {confidence:.1%}",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=confidence_color
        ).pack()
        
        # Content
        content_frame = ctk.CTkFrame(result_card, fg_color="transparent")
        content_frame.pack(fill="x", padx=15, pady=(0, 15))
        
        # Summary
        if 'summary' in result_data:
            ctk.CTkLabel(
                content_frame,
                text="Summary:",
                font=ctk.CTkFont(size=12, weight="bold"),
                anchor="w"
            ).pack(fill="x", pady=(0, 5))
            
            summary_text = ctk.CTkTextbox(content_frame, height=60)
            summary_text.pack(fill="x", pady=(0, 10))
            summary_text.insert("1.0", result_data['summary'])
            summary_text.configure(state="disabled")
        
        # Key points
        if 'key_points' in result_data:
            ctk.CTkLabel(
                content_frame,
                text="Key Points:",
                font=ctk.CTkFont(size=12, weight="bold"),
                anchor="w"
            ).pack(fill="x", pady=(0, 5))
            
            for point in result_data['key_points'][:5]:
                point_frame = ctk.CTkFrame(content_frame, fg_color="transparent")
                point_frame.pack(fill="x", pady=2)
                
                ctk.CTkLabel(
                    point_frame,
                    text="•",
                    font=ctk.CTkFont(size=12),
                    width=20
                ).pack(side="left")
                
                ctk.CTkLabel(
                    point_frame,
                    text=point,
                    font=ctk.CTkFont(size=12),
                    anchor="w",
                    justify="left"
                ).pack(side="left", fill="x", expand=True)
        
        # Detail Analysis Button
        detail_btn_frame = ctk.CTkFrame(content_frame, fg_color="transparent")
        detail_btn_frame.pack(fill="x", pady=(10, 0))
        
        detail_btn = ctk.CTkButton(
            detail_btn_frame,
            text="🔍 Detailed Analysis",
            command=lambda fp=file_path, rd=result_data: self.open_detailed_analysis(fp, rd),
            fg_color="#9C27B0",
            hover_color="#7B1FA2",
            height=35
        )
        detail_btn.pack(side="right")
                
    def open_detailed_analysis(self, file_path: str, result_data: Dict):
        """Open detailed analysis tab for the document"""
        self.parent.create_detailed_analysis_tab(file_path, result_data)
                
    def clear_results(self):
        """Clear all results"""
        for widget in self.results_frame.winfo_children():
            widget.destroy()


class RightPanel:
    """Right panel with document actions"""
    
    def __init__(self, parent, main_content):
        self.parent = parent
        self.main_content = main_content
        
        # Create right panel frame
        self.right_panel = ctk.CTkFrame(main_content, width=250)
        self.right_panel.grid(row=0, column=1, rowspan=2, sticky="nsew", padx=(5, 0), pady=5)
        self.right_panel.grid_propagate(False)
        
        # Configure grid
        self.right_panel.grid_rowconfigure(2, weight=1)
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup right panel UI"""
        # Header
        header_frame = ctk.CTkFrame(self.right_panel, fg_color="transparent", height=40)
        header_frame.grid(row=0, column=0, sticky="ew", padx=15, pady=15)
        header_frame.grid_propagate(False)
        
        self.header_label = ctk.CTkLabel(
            header_frame,
            text="Document Actions",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.header_label.pack(anchor="w")
        
        # Separator
        separator = ctk.CTkFrame(self.right_panel, height=2, fg_color="#3b3b3b")
        separator.grid(row=1, column=0, sticky="ew", padx=10, pady=5)
        
        # Actions container
        self.actions_frame = ctk.CTkFrame(self.right_panel, fg_color="transparent")
        self.actions_frame.grid(row=2, column=0, sticky="nsew", padx=10, pady=10)
        self.right_panel.grid_rowconfigure(2, weight=1)
        
        self.setup_document_actions()
        
    def setup_document_actions(self):
        """Setup document action buttons"""
        # Clear existing buttons
        for widget in self.actions_frame.winfo_children():
            widget.destroy()
            
        buttons_config = [
            ("📁 Add Document", self.add_document, "#4CC9F0"),
            ("✓ Select All", self.select_all, "#2196F3"),
            ("✗ Deselect All", self.deselect_all, "#2196F3"),
            ("🔍 Analyze Selected", self.analyze_selected, "#4CAF50"),
            ("🗑️ Delete Selected", self.delete_selected, "#F44336"),
            ("📂 Categorize & Save", self.categorize_save, "#FF9800"),
            ("📊 Quick Analysis", self.quick_analysis, "#9C27B0"),
            ("🔄 Refresh View", self.refresh_view, "#607D8B"),
        ]
        
        for text, command, color in buttons_config:
            btn = ctk.CTkButton(
                self.actions_frame,
                text=text,
                command=command,
                fg_color=color,
                hover_color=color,
                height=40,
                anchor="w",
                font=ctk.CTkFont(size=13)
            )
            btn.pack(fill="x", pady=5)
            
        # Add spacing
        ctk.CTkFrame(self.actions_frame, height=20, fg_color="transparent").pack()
        
    def setup_detail_analysis_actions(self):
        """Setup detail analysis action buttons"""
        # Clear existing buttons
        for widget in self.actions_frame.winfo_children():
            widget.destroy()
            
        buttons_config = [
            ("📖 Switch to Hierarchical", self.switch_hierarchical, "#4CC9F0"),
            ("📋 Switch to Point-wise", self.switch_pointwise, "#2196F3"),
            ("💬 Clear Chat", self.clear_chat, "#4CAF50"),
            ("📥 Export Analysis", self.export_analysis, "#FF9800"),
            ("🔄 Refresh Data", self.refresh_data, "#607D8B"),
        ]
        
        for text, command, color in buttons_config:
            btn = ctk.CTkButton(
                self.actions_frame,
                text=text,
                command=command,
                fg_color=color,
                hover_color=color,
                height=40,
                anchor="w",
                font=ctk.CTkFont(size=13)
            )
            btn.pack(fill="x", pady=5)
            
        # Add spacing
        ctk.CTkFrame(self.actions_frame, height=20, fg_color="transparent").pack()
        
    def switch_hierarchical(self):
        """Switch to hierarchical view in detail analysis"""
        current_tab = self.parent.get_current_tab()
        if hasattr(current_tab, 'view_var'):
            current_tab.view_var.set("hierarchical")
            current_tab.toggle_view()
            
    def switch_pointwise(self):
        """Switch to point-wise view in detail analysis"""
        current_tab = self.parent.get_current_tab()
        if hasattr(current_tab, 'view_var'):
            current_tab.view_var.set("pointwise")
            current_tab.toggle_view()
            
    def clear_chat(self):
        """Clear chat history in detail analysis"""
        current_tab = self.parent.get_current_tab()
        if hasattr(current_tab, 'chat_history_frame'):
            # Clear chat history UI
            for widget in current_tab.chat_history_frame.winfo_children():
                widget.destroy()
            # Clear chat history data
            current_tab.chat_history.clear()
            # Add welcome message back
            current_tab.add_chat_message("assistant", "Hello! I can answer questions about this document. What would you like to know?")
            
    def export_analysis(self):
        """Export analysis data"""
        current_tab = self.parent.get_current_tab()
        if hasattr(current_tab, 'analysis_data'):
            file_path = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            if file_path:
                try:
                    with open(file_path, 'w') as f:
                        json.dump(current_tab.analysis_data, f, indent=2)
                    self.parent.show_message("Export Successful", f"Analysis data exported to {file_path}")
                except Exception as e:
                    self.parent.show_error("Export Error", f"Failed to export: {str(e)}")
                    
    def refresh_data(self):
        """Refresh analysis data"""
        self.parent.show_message("Refresh", "Analysis data refresh would be implemented here")
        
    def update_for_tab_type(self, tab_type: str):
        """Update right panel based on current tab type"""
        if tab_type == "detail_analysis":
            self.header_label.configure(text="Detail Analysis")
            self.setup_detail_analysis_actions()
        else:
            self.header_label.configure(text="Document Actions")
            self.setup_document_actions()
        
    def add_document(self):
        """Add document to current tab"""
        current_tab = self.parent.get_current_document_tab()
        if current_tab:
            current_tab.browse_files()
            
    def select_all(self):
        """Select all documents in current tab"""
        current_tab = self.parent.get_current_document_tab()
        if current_tab:
            current_tab.select_all_documents()
            
    def deselect_all(self):
        """Deselect all documents in current tab"""
        current_tab = self.parent.get_current_document_tab()
        if current_tab:
            current_tab.deselect_all_documents()
            
    def analyze_selected(self):
        """Analyze selected documents"""
        current_tab = self.parent.get_current_document_tab()
        if current_tab:
            selected = current_tab.get_selected_documents()
            if selected:
                self.parent.analyze_documents(selected, current_tab)
            else:
                self.parent.show_message("No Selection", "Please select documents to analyze")
                
    def delete_selected(self):
        """Delete selected documents"""
        current_tab = self.parent.get_current_document_tab()
        if current_tab:
            selected = current_tab.get_selected_documents()
            if selected:
                if self.parent.ask_yes_no("Confirm Delete", f"Delete {len(selected)} selected documents?"):
                    for file_path in selected:
                        if file_path in current_tab.documents:
                            del current_tab.documents[file_path]
                    current_tab.update_view()
                    self.parent.update_dashboard_stats()
                    self.parent.update_button_states()
                    
    def categorize_save(self):
        """Categorize and save documents"""
        current_tab = self.parent.get_current_document_tab()
        if current_tab:
            selected = current_tab.get_selected_documents()
            if selected:
                self.parent.show_message("Categorize & Save", f"Will categorize and save {len(selected)} documents")
            else:
                self.parent.show_warning("No Selection", "Please select documents to categorize")
                
    def quick_analysis(self):
        """Quick analysis of selected documents"""
        current_tab = self.parent.get_current_document_tab()
        if current_tab:
            selected = current_tab.get_selected_documents()
            if selected:
                self.parent.show_message("Quick Analysis", f"Will perform quick analysis on {len(selected)} documents")
            else:
                self.parent.show_warning("No Selection", "Please select documents to analyze")
                
    def refresh_view(self):
        """Refresh the current view"""
        current_tab = self.parent.get_current_document_tab()
        if current_tab:
            current_tab.update_view()


class DocuMindApp:
    """Main DocuMind AI Application"""
    
    def __init__(self):
        # Initialize components
        self.root = None
        self.ocr_processor = None
        self.batch_analyzer = None
        self.document_tabs: Dict[str, DocumentTab] = {}
        self.current_tabs: List[DocumentTab] = {}
        self.detail_analysis_tabs: Dict[str, DetailAnalysisTab] = {}
        
        # GUI management
        self.gui_manager = None
        self.splash = None
        
        # Start initialization
        self.initialize_app()
        
    def initialize_app(self):
        """Initialize application components"""
        # Create root window first (hidden)
        self.root = ctk.CTk()
        self.root.title("DocuMind AI")
        self.root.geometry("1400x900")
        self.root.withdraw()
        
        # Setup GUI manager
        self.gui_manager = ThreadSafeGUI(self.root)
        
        # Show splash screen
        self.splash = SplashScreen(self.gui_manager)
        self.splash.show()
        
        # Start task processing
        self.gui_manager.process_tasks()
        
        # Initialize in background
        threading.Thread(target=self.initialize_backend, daemon=True).start()
        
    def initialize_backend(self):
        """Initialize backend components in background thread"""
        try:
            # Simulate initialization steps
            steps = [
                ("Loading OCR Engine...", 0.2),
                ("Initializing AI Models...", 0.4),
                ("Setting up Document Processor...", 0.6),
                ("Preparing User Interface...", 0.8),
                ("Ready to use!", 1.0)
            ]
            
            for status, progress in steps:
                self.splash.update_status(status, progress)
                time.sleep(0.5)
                
            # Initialize actual components
            self.splash.update_status("Initializing OCR Engine...", 0.3)
            poppler_path = r"C:\poppler-windows\poppler-25.07.0\Library\bin"
            self.ocr_processor = OCRProcessor(
                enable_realtime_analysis=True,
                poppler_path=poppler_path
            )
            
            self.splash.update_status("Loading AI Models...", 0.7)
            api_key = "AIzaSyABE4KS1xFosi0M6gFkpLGt8gVNZE5Haa8"
            self.batch_analyzer = SmartBatchAnalyzer(api_key=api_key)
            
            # Setup callbacks
            self.setup_callbacks()
            
            # Close splash and show main window
            time.sleep(0.5)
            self.gui_manager.schedule_task(self.show_main_window)
            
        except Exception as e:
            self.gui_manager.schedule_task(
                self.show_error, 
                "Initialization Error", 
                f"Failed to initialize: {str(e)}"
            )
            
    def show_main_window(self):
        """Show the main application window"""
        self.splash.close()
        self.setup_main_ui()
        self.root.deiconify()
        self.root.focus_force()
        
    def setup_main_ui(self):
        """Setup main application UI"""
        # Configure theme
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")
        
        # Configure grid layout
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)
        
        # Create sidebar
        self.setup_sidebar()
        
        # Create main content area
        self.setup_main_content()
        
        # Bind tab change event
        self.tab_control.configure(command=self.on_tab_changed)
        
    def setup_sidebar(self):
        """Setup sidebar dashboard"""
        self.sidebar = ctk.CTkFrame(self.root, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_rowconfigure(8, weight=1)
        
        # Logo and title
        logo_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        logo_frame.grid(row=0, column=0, padx=20, pady=20, sticky="ew")
        
        ctk.CTkLabel(
            logo_frame,
            text="📄 DocuMind AI",
            font=ctk.CTkFont(size=20, weight="bold"),
            anchor="w"
        ).pack(fill="x")
        
        ctk.CTkLabel(
            logo_frame,
            text="Document Intelligence",
            font=ctk.CTkFont(size=12),
            text_color="#B0B0B0",
            anchor="w"
        ).pack(fill="x")
        
        # Navigation buttons
        nav_buttons = [
            ("📊 Dashboard", self.show_dashboard),
            ("📁 Documents", self.show_documents),
            ("🔍 Analysis", self.show_analysis),
            ("⚙️ Settings", self.show_settings)
        ]
        
        for i, (text, command) in enumerate(nav_buttons, 1):
            btn = ctk.CTkButton(
                self.sidebar,
                text=text,
                command=command,
                fg_color="transparent",
                hover_color="#3b3b3b",
                anchor="w",
                height=40
            )
            btn.grid(row=i, column=0, padx=10, pady=5, sticky="ew")
        
        # Statistics section
        stats_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        stats_frame.grid(row=9, column=0, padx=20, pady=20, sticky="ew")
        
        ctk.CTkLabel(
            stats_frame,
            text="Statistics",
            font=ctk.CTkFont(size=14, weight="bold"),
            anchor="w"
        ).pack(fill="x", pady=(0, 10))
        
        self.stats_labels = {}
        stats_data = [
            ("Total Documents", "0"),
            ("Processed", "0"),
            ("Categories", "0"),
            ("Analysis Time", "0s")
        ]
        
        for label, value in stats_data:
            stat_frame = ctk.CTkFrame(stats_frame, fg_color="transparent")
            stat_frame.pack(fill="x", pady=2)
            
            ctk.CTkLabel(
                stat_frame,
                text=label,
                font=ctk.CTkFont(size=11),
                text_color="#B0B0B0",
                anchor="w"
            ).pack(side="left")
            
            self.stats_labels[label] = ctk.CTkLabel(
                stat_frame,
                text=value,
                font=ctk.CTkFont(size=11, weight="bold"),
                anchor="e"
            )
            self.stats_labels[label].pack(side="right")
        
    def setup_main_content(self):
        """Setup main content area"""
        # Main content frame
        main_content = ctk.CTkFrame(self.root)
        main_content.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        
        # Configure grid weights
        main_content.grid_rowconfigure(1, weight=1)
        main_content.grid_columnconfigure(0, weight=1)
        main_content.grid_columnconfigure(1, weight=0)
        
        # Header frame
        header_frame = ctk.CTkFrame(main_content, fg_color="transparent", height=35)
        header_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=(5, 0))
        header_frame.grid_propagate(False)
        
        # Header content
        header_content = ctk.CTkFrame(header_frame, fg_color="transparent")
        header_content.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(
            header_content,
            text="Document Workspace",
            font=ctk.CTkFont(size=12, weight="bold")
        ).pack(side="left")
        
        new_tab_btn = ctk.CTkButton(
            header_content,
            text="+ New Tab",
            command=self.create_new_tab,
            width=70,
            height=25,
            fg_color="#4CC9F0",
            hover_color="#3AA8D9"
        )
        new_tab_btn.pack(side="right")
        
        # Tab control
        self.tab_control = ctk.CTkTabview(main_content)
        self.tab_control.grid(row=1, column=0, sticky="nsew", padx=5, pady=(0, 5))
        
        # Right panel
        self.right_panel = RightPanel(self, main_content)
        
        # Create initial tabs
        self.create_new_tab("Documents")
        self.analysis_tab = AnalysisResultsTab(self, self.tab_control)
        
    def on_tab_changed(self):
        """Handle tab change event"""
        current_tab = self.get_current_tab()
        if hasattr(current_tab, 'tab_name') and current_tab.tab_name.startswith("Detail:"):
            self.right_panel.update_for_tab_type("detail_analysis")
        else:
            self.right_panel.update_for_tab_type("document")
            
    def create_new_tab(self, tab_name=None):
        """Create a new document tab"""
        if tab_name is None:
            tab_count = len(self.document_tabs) + 1
            tab_name = f"Tab {tab_count}"
            
        # Check if tab name already exists
        if tab_name in self.document_tabs:
            return self.document_tabs[tab_name]
            
        try:
            new_tab = DocumentTab(self, self.tab_control, tab_name)
            self.document_tabs[tab_name] = new_tab
            self.current_tabs[tab_name] = new_tab
            
            # Switch to the new tab
            self.tab_control.set(tab_name)
            
            return new_tab
            
        except Exception as e:
            self.show_error("Tab Creation Error", f"Failed to create tab: {str(e)}")
            return None
            
    def create_detailed_analysis_tab(self, file_path: str, analysis_data: Dict):
        """Create a detailed analysis tab for a document"""
        tab_name = f"Detail: {Path(file_path).name}"
        
        # Check if tab already exists
        if tab_name in self.detail_analysis_tabs:
            self.tab_control.set(tab_name)
            return self.detail_analysis_tabs[tab_name]
            
        try:
            detail_tab = DetailAnalysisTab(self, self.tab_control, file_path, analysis_data)
            self.detail_analysis_tabs[tab_name] = detail_tab
            self.current_tabs[tab_name] = detail_tab
            
            # Switch to the new tab
            self.tab_control.set(tab_name)
            
            # Update right panel for detail analysis
            self.right_panel.update_for_tab_type("detail_analysis")
            
            return detail_tab
            
        except Exception as e:
            self.show_error("Detail Analysis Error", f"Failed to create detail analysis tab: {str(e)}")
            return None
            
    def get_current_tab(self):
        """Get currently active tab of any type"""
        current_tab_name = self.tab_control.get()
        return self.current_tabs.get(current_tab_name)
        
    def get_current_document_tab(self):
        """Get currently active document tab"""
        current_tab_name = self.tab_control.get()
        return self.document_tabs.get(current_tab_name)
        
    def rename_document_tab(self, old_name: str, new_name: str):
        """Rename document tab"""
        if old_name in self.document_tabs:
            tab = self.document_tabs.pop(old_name)
            self.document_tabs[new_name] = tab
            self.current_tabs[new_name] = tab
            
    def update_dashboard_stats(self):
        """Update dashboard statistics"""
        total_docs = sum(len(tab.documents) for tab in self.document_tabs.values())
        processed_docs = 0  # Track this properly
        
        if "Total Documents" in self.stats_labels:
            self.stats_labels["Total Documents"].configure(text=str(total_docs))
        if "Processed" in self.stats_labels:
            self.stats_labels["Processed"].configure(text=str(processed_docs))
            
    def update_button_states(self):
        """Update button states based on current selection"""
        # Implement based on your needs
        pass
        
    def setup_callbacks(self):
        """Setup callbacks for OCR and AI processors"""
        if self.ocr_processor:
            self.ocr_processor.set_analysis_callback(self.on_page_processed)
            
        if self.batch_analyzer:
            self.batch_analyzer.set_batch_analysis_callback(self.on_batch_analyzed)
            self.batch_analyzer.set_final_analysis_callback(self.on_final_analysis)
            
    def on_page_processed(self, file_path: str, page_num: int, ocr_result):
        """Called when OCR processes a page"""
        self.gui_manager.schedule_task(
            self.update_processing_status, 
            file_path, 
            f"Page {page_num} processed"
        )
        
        if self.batch_analyzer:
            self.batch_analyzer.add_page_result(file_path, page_num, ocr_result)
        
    def on_batch_analyzed(self, batch_result: BatchAnalysisResult):
        """Called when batch analysis is complete"""
        self.gui_manager.schedule_task(self.display_batch_result, batch_result)
        
    def on_final_analysis(self, final_result: BatchAnalysisResult):
        """Called when final analysis is complete"""
        self.gui_manager.schedule_task(self.display_final_result, final_result)
        
    def update_processing_status(self, file_path: str, status: str):
        """Update processing status across all tabs"""
        for tab in self.document_tabs.values():
            tab.update_document_status(file_path, status)
            
    def display_batch_result(self, batch_result: BatchAnalysisResult):
        """Display batch analysis result"""
        file_path = batch_result.file_path
        result_data = batch_result.result
        
        self.update_processing_status(file_path, "Analyzed")
        
        if self.analysis_tab:
            self.analysis_tab.add_result(file_path, result_data)
            
        self.update_dashboard_stats()
        
    def display_final_result(self, final_result: BatchAnalysisResult):
        """Display final analysis result"""
        file_path = final_result.file_path
        result_data = final_result.result
        
        self.update_processing_status(file_path, "Complete")
        
        self.show_message(
            "Analysis Complete",
            f"Analysis completed for {Path(file_path).name}\n"
            f"Category: {result_data.get('document_category', 'Unknown')}"
        )
        
    def analyze_documents(self, documents: List[str], source_tab):
        """Start analysis for selected documents"""
        if not documents:
            self.show_warning("No Selection", "Please select documents to analyze")
            return
            
        # Update status for selected documents
        for doc_path in documents:
            source_tab.update_document_status(doc_path, "Processing...")
            
        # Start processing in background
        def process_documents():
            for doc_path in documents:
                try:
                    if self.batch_analyzer:
                        total_pages = self.estimate_total_pages(doc_path)
                        self.batch_analyzer.register_document(doc_path, total_pages)
                        
                    if self.ocr_processor:
                        self.ocr_processor.process_document_async(doc_path)
                        
                except Exception as e:
                    self.gui_manager.schedule_task(
                        self.update_processing_status, 
                        doc_path, 
                        f"Error: {str(e)}"
                    )
                    
        threading.Thread(target=process_documents, daemon=True).start()
        
    def estimate_total_pages(self, file_path: str) -> int:
        """Estimate total pages in document"""
        # Implement proper PDF page counting
        return 10
        
    # Utility methods for thread-safe dialogs
    def show_message(self, title: str, message: str):
        """Show info message"""
        self.gui_manager.schedule_task(
            lambda: messagebox.showinfo(title, message)
        )
        
    def show_warning(self, title: str, message: str):
        """Show warning message"""
        self.gui_manager.schedule_task(
            lambda: messagebox.showwarning(title, message)
        )
        
    def show_error(self, title: str, message: str):
        """Show error message"""
        self.gui_manager.schedule_task(
            lambda: messagebox.showerror(title, message)
        )
        
    def ask_yes_no(self, title: str, message: str) -> bool:
        """Ask yes/no question (not thread-safe)"""
        return messagebox.askyesno(title, message)
        
    # Navigation methods
    def show_dashboard(self):
        """Show dashboard view"""
        self.tab_control.set("Analysis Results")
        
    def show_documents(self):
        """Show documents view"""
        if self.document_tabs:
            first_tab = next(iter(self.document_tabs.values()))
            self.tab_control.set(first_tab.tab_name)
            
    def show_analysis(self):
        """Show analysis view"""
        self.tab_control.set("Analysis Results")
        
    def show_settings(self):
        """Show settings dialog"""
        self.show_message("Settings", "Settings dialog will be implemented here")
        
    def run(self):
        """Start the application"""
        self.root.mainloop()


# Main entry point
if __name__ == "__main__":
    app = DocuMindApp()
    app.run()