export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
    public code?: string,
    public details?: Record<string, unknown>,
  ) {
    super(message);
  }
}

export async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`/api${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
    signal: init?.signal ?? AbortSignal.timeout(60000),
  });
  const body = await response.json().catch(() => null);
  if (!response.ok) {
    const detail = body?.detail;
    const message = body?.error?.message
      ? `${body.error.code}: ${body.error.field_path ? body.error.field_path + ": " : ""}${body.error.message}`
      : typeof detail === "string"
        ? detail
        : Array.isArray(detail)
          ? detail
              .map(
                (e: { loc?: string[]; msg: string }) =>
                  `${e.loc?.slice(1).join(".") ?? "Input"}: ${e.msg}`,
              )
              .join("; ")
          : `Request failed (${response.status})`;
    throw new ApiError(message, response.status, body?.error?.code, body?.error?.details);
  }
  return body as T;
}

export const post = <T>(path: string, body?: unknown) =>
  request<T>(path, {
    method: "POST",
    body: body === undefined ? undefined : JSON.stringify(body),
  });
