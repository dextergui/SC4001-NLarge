import { redirect } from "next/navigation";

export default function DocumentationMain() {
  redirect("/documentation/installation");
  return null;
}
