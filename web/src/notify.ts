const STORAGE_KEY = "luxaeterna_notify_results";

export function isNotificationApiAvailable(): boolean {
  return typeof window !== "undefined" && "Notification" in window;
}

export function getNotifyPreference(): boolean {
  try {
    return localStorage.getItem(STORAGE_KEY) === "1";
  } catch {
    return false;
  }
}

export function setNotifyPreference(on: boolean): void {
  try {
    if (on) localStorage.setItem(STORAGE_KEY, "1");
    else localStorage.removeItem(STORAGE_KEY);
  } catch {
    /* ignore */
  }
}

export function notificationPermission(): NotificationPermission | "unsupported" {
  if (!isNotificationApiAvailable()) return "unsupported";
  return Notification.permission;
}

/**
 * Must be called from a user gesture for best browser support.
 */
export async function requestNotificationPermission(): Promise<NotificationPermission | "unsupported"> {
  if (!isNotificationApiAvailable()) return "unsupported";
  try {
    return await Notification.requestPermission();
  } catch {
    return Notification.permission;
  }
}

export function notifyResult(title: string, body: string, tag = "luxaeterna-result"): void {
  if (!isNotificationApiAvailable() || Notification.permission !== "granted") return;
  if (!getNotifyPreference()) return;
  try {
    new Notification(title, {
      body,
      tag,
      silent: false,
    });
  } catch {
    /* ignore */
  }
}
