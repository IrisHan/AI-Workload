import React from "react";
import ReactDOM from "react-dom/client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { Layout } from "./app/Layout";
import { ImportModelPage, ModelDetailPage, ModelsPage } from "./pages/models";
import { StudioFocusPage } from "./pages/studio-focus";
import {
  NewWorkloadPage,
  WorkloadDetailPage,
  WorkloadsPage,
  InferenceDesignPage,
  WorkloadComparePage,
} from "./pages/workloads";
import {
  AdministrationPage,
  DashboardPage,
  FuturePage,
} from "./pages/catalog";
import { HardwarePage, HardwareDetailPage } from "./pages/hardware";
import "./styles.css";

const client = new QueryClient({
  defaultOptions: {
    queries: { retry: 1, staleTime: 30000, refetchOnWindowFocus: false },
  },
});
ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <QueryClientProvider client={client}>
      <BrowserRouter>
        <Routes>
          <Route element={<Layout />}>
            <Route index element={<Navigate to="/models" replace />} />
            <Route path="dashboard" element={<DashboardPage />} />
            <Route path="models" element={<ModelsPage />} />
            <Route path="models/import" element={<ImportModelPage />} />
            <Route path="models/:id" element={<ModelDetailPage />} />
            <Route path="models/:id/studio" element={<StudioFocusPage />} />
            <Route path="workloads" element={<WorkloadsPage />} />
            <Route path="workloads/new" element={<NewWorkloadPage />} />
            <Route path="workloads/inference/new" element={<InferenceDesignPage />} />
            <Route path="workloads/inference/:designId" element={<InferenceDesignPage />} />
            <Route path="workloads/compare" element={<WorkloadComparePage />} />
            <Route path="workloads/:id" element={<WorkloadDetailPage />} />
            <Route path="hardware" element={<HardwarePage />} />
            <Route path="hardware/:id" element={<HardwareDetailPage />} />
            <Route path="administration" element={<AdministrationPage />} />
            <Route path="*" element={<FuturePage />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  </React.StrictMode>,
);
