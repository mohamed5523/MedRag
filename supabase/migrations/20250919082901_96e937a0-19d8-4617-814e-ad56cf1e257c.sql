-- Fix security issues: Enable RLS and create policies

-- Enable RLS on all new tables
ALTER TABLE public.documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.doctors ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.doctor_schedules ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.appointments ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.chat_logs ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.document_embeddings ENABLE ROW LEVEL SECURITY;

-- Documents policies
CREATE POLICY "Managers can manage all documents" ON public.documents
FOR ALL USING (
  EXISTS (SELECT 1 FROM public.profiles WHERE id = auth.uid() AND role = 'manager')
);

CREATE POLICY "Staff can view documents" ON public.documents
FOR SELECT USING (
  EXISTS (SELECT 1 FROM public.profiles WHERE id = auth.uid() AND role IN ('staff', 'manager'))
);

-- Storage policies for documents
CREATE POLICY "Managers can manage document files" ON storage.objects
FOR ALL USING (bucket_id = 'documents' AND 
  EXISTS (SELECT 1 FROM public.profiles WHERE id = auth.uid() AND role = 'manager'));

CREATE POLICY "Staff can view document files" ON storage.objects
FOR SELECT USING (bucket_id = 'documents' AND 
  EXISTS (SELECT 1 FROM public.profiles WHERE id = auth.uid() AND role IN ('staff', 'manager')));

-- Doctors policies - everyone can view, only managers can modify
CREATE POLICY "Everyone can view doctors" ON public.doctors FOR SELECT USING (true);
CREATE POLICY "Managers can manage doctors" ON public.doctors
FOR INSERT USING (
  EXISTS (SELECT 1 FROM public.profiles WHERE id = auth.uid() AND role = 'manager')
);
CREATE POLICY "Managers can update doctors" ON public.doctors
FOR UPDATE USING (
  EXISTS (SELECT 1 FROM public.profiles WHERE id = auth.uid() AND role = 'manager')
);
CREATE POLICY "Managers can delete doctors" ON public.doctors
FOR DELETE USING (
  EXISTS (SELECT 1 FROM public.profiles WHERE id = auth.uid() AND role = 'manager')
);

-- Doctor schedules - everyone can view
CREATE POLICY "Everyone can view schedules" ON public.doctor_schedules FOR SELECT USING (true);
CREATE POLICY "Managers can manage schedules" ON public.doctor_schedules
FOR INSERT USING (
  EXISTS (SELECT 1 FROM public.profiles WHERE id = auth.uid() AND role = 'manager')
);
CREATE POLICY "Managers can update schedules" ON public.doctor_schedules
FOR UPDATE USING (
  EXISTS (SELECT 1 FROM public.profiles WHERE id = auth.uid() AND role = 'manager')
);
CREATE POLICY "Managers can delete schedules" ON public.doctor_schedules
FOR DELETE USING (
  EXISTS (SELECT 1 FROM public.profiles WHERE id = auth.uid() AND role = 'manager')
);

-- Appointments policies
CREATE POLICY "Patients can view own appointments" ON public.appointments
FOR SELECT USING (
  patient_id = auth.uid() OR 
  EXISTS (SELECT 1 FROM public.profiles WHERE id = auth.uid() AND role IN ('staff', 'manager'))
);

CREATE POLICY "Patients can create own appointments" ON public.appointments
FOR INSERT WITH CHECK (patient_id = auth.uid());

CREATE POLICY "Staff can manage appointments" ON public.appointments
FOR UPDATE USING (
  EXISTS (SELECT 1 FROM public.profiles WHERE id = auth.uid() AND role IN ('staff', 'manager'))
);
CREATE POLICY "Staff can delete appointments" ON public.appointments
FOR DELETE USING (
  EXISTS (SELECT 1 FROM public.profiles WHERE id = auth.uid() AND role IN ('staff', 'manager'))
);

-- Chat logs policies
CREATE POLICY "Users can view own chat logs" ON public.chat_logs
FOR SELECT USING (
  user_id = auth.uid() OR 
  EXISTS (SELECT 1 FROM public.profiles WHERE id = auth.uid() AND role = 'manager')
);

CREATE POLICY "Users can create own chat logs" ON public.chat_logs
FOR INSERT WITH CHECK (user_id = auth.uid());

-- Document embeddings - staff and managers can view for RAG queries
CREATE POLICY "Staff can view embeddings" ON public.document_embeddings
FOR SELECT USING (
  EXISTS (SELECT 1 FROM public.profiles WHERE id = auth.uid() AND role IN ('staff', 'manager'))
);

CREATE POLICY "Managers can manage embeddings" ON public.document_embeddings
FOR INSERT USING (
  EXISTS (SELECT 1 FROM public.profiles WHERE id = auth.uid() AND role = 'manager')
);
CREATE POLICY "Managers can update embeddings" ON public.document_embeddings
FOR UPDATE USING (
  EXISTS (SELECT 1 FROM public.profiles WHERE id = auth.uid() AND role = 'manager')
);
CREATE POLICY "Managers can delete embeddings" ON public.document_embeddings
FOR DELETE USING (
  EXISTS (SELECT 1 FROM public.profiles WHERE id = auth.uid() AND role = 'manager')
);

-- Create triggers for updated_at columns
DROP TRIGGER IF EXISTS update_documents_updated_at ON public.documents;
CREATE TRIGGER update_documents_updated_at BEFORE UPDATE ON public.documents
    FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();

DROP TRIGGER IF EXISTS update_doctors_updated_at ON public.doctors;
CREATE TRIGGER update_doctors_updated_at BEFORE UPDATE ON public.doctors
    FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();

DROP TRIGGER IF EXISTS update_appointments_updated_at ON public.appointments;
CREATE TRIGGER update_appointments_updated_at BEFORE UPDATE ON public.appointments
    FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();