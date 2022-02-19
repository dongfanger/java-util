import java.util.*;

/**
 * 100道算法题 Algorithm Question
 */
public class AQ100 {
    //======================简单题开始======================
    //1大数加法
    public String bigNumberAdd(String s, String t) {
        // write code here
        Stack<Integer> stack = new Stack<>();
        StringBuilder result = new StringBuilder();
        int i = s.length() - 1;
        int j = t.length() - 1;
        int carry = 0;
        while (i >= 0 || j >= 0 || carry != 0) {
            carry += i >= 0 ? s.charAt(i--) - '0' : 0;
            carry += j >= 0 ? t.charAt(j--) - '0' : 0;
            stack.push(carry % 10);
            carry /= 10;
        }
        while (!stack.isEmpty()) {
            result.append(stack.pop());
        }
        return result.toString();
    }

    //2链表中环的入口结点
    public static class ListNode {
        int val;
        ListNode next = null;

        ListNode(int val) {
            this.val = val;
        }
    }

    public ListNode EntryNodeOfLoop(ListNode pHead) {
        if (pHead == null) return null;
        ListNode slow = pHead;
        ListNode fast = pHead;
        while (fast != null && fast.next != null) {
            fast = fast.next.next;
            slow = slow.next;
            if (slow == fast) {
                break;
            }
        }

        if (fast == null || fast.next == null) {
            return null;
        }

        fast = pHead;

        while (fast != slow) {
            fast = fast.next;
            slow = slow.next;
        }

        return fast;

    }

    //3判断链表中是否有环
    public boolean hasCycle(ListNode head) {
        ListNode pos = head;
        Set<ListNode> visited = new HashSet<>();
        while (pos != null) {
            if (visited.contains(pos)) {
                return true;
            } else {
                visited.add(pos);
            }
            pos = pos.next;
        }
        return false;
    }

    //4二叉树中的最大路径和
    public static class TreeNode {
        int val = 0;
        TreeNode left = null;
        TreeNode right = null;

        public TreeNode(int i) {
            val = i;
        }
    }

    int maxSum = Integer.MIN_VALUE;

    public int maxPathSum(TreeNode root) {
        // write code here
        maxGain(root);
        return maxSum;
    }

    public int maxGain(TreeNode node) {
        if (node == null) {
            return 0;
        }

        int leftGain = Math.max(maxGain(node.left), 0);
        int rightGain = Math.max(maxGain(node.right), 0);

        int priceNewPath = node.val + leftGain + rightGain;
        maxSum = Math.max(maxSum, priceNewPath);

        return node.val + Math.max(leftGain, rightGain);
    }

    //5将升序数组转化为平衡二叉搜索树
    public TreeNode sortedArrayToBST(int[] num) {
        // write code here
        if (num == null || num.length == 0) {
            return null;
        }

        return process(num, 0, num.length - 1);
    }

    public TreeNode process(int[] num, int left, int right) {
        if (left > right) {
            return null;
        }

        if (left == right) {
            return new TreeNode(num[left]);
        }

        int len = right - left + 1;
        int mid = left + len / 2;
        TreeNode root = new TreeNode(num[mid]);
        root.left = process(num, left, mid - 1);
        root.right = process(num, mid + 1, right);
        return root;
    }

    //6重建二叉树
    public TreeNode reConstructBinaryTree(int[] pre, int[] vin) {
        return reConstructBinaryTree(pre, 0, pre.length - 1, vin, 0, vin.length - 1);
    }

    private TreeNode reConstructBinaryTree(int[] pre, int startPre, int endPre, int[] vin, int startVin, int endVin) {
        if (startPre > endPre || startVin > endVin) {
            return null;
        }

        TreeNode root = new TreeNode(pre[startPre]);

        for (int i = startVin; i <= endVin; i++) {
            if (vin[i] == pre[startPre]) {
                root.left = reConstructBinaryTree(pre, startPre + 1, startPre + i - startVin, vin, startVin, i - 1);
                root.right = reConstructBinaryTree(pre, i - startVin + startPre + 1, endPre, vin, i + 1, endVin);
                break;
            }
        }

        return root;
    }

    //7按之字形顺序打印二叉树
    public ArrayList<ArrayList<Integer>> Print(TreeNode pRoot) {
        LinkedList<TreeNode> deque = new LinkedList<>();
        ArrayList res = new ArrayList<>();
        if (pRoot != null) {
            deque.add(pRoot);
        }
        while (!deque.isEmpty()) {
            ArrayList tmp = new ArrayList<>();
            for (int i = deque.size(); i > 0; i--) {
                TreeNode node = deque.removeFirst();
                tmp.add(node.val);
                if (node.left != null) {
                    deque.addLast(node.left);
                }
                if (node.right != null) {
                    deque.addLast(node.right);
                }
            }
            res.add(tmp);
            if (deque.isEmpty()) {
                break;
            }
            tmp = new ArrayList<>();
            for (int i = deque.size(); i > 0; i--) {
                TreeNode node = deque.removeLast();
                tmp.add(node.val);
                if (node.right != null) {
                    deque.addFirst(node.right);
                }
                if (node.left != null) {
                    deque.addFirst(node.left);
                }
            }
            res.add(tmp);
        }

        return res;
    }

    //8求二叉树的层序遍历
    public ArrayList<ArrayList<Integer>> levelOrder(TreeNode root) {
        // write code here
        ArrayList<ArrayList<Integer>> res = new ArrayList<>();
        if (root == null) {
            return res;
        }

        dfs(root, res, 0);
        return res;
    }

    public void dfs(TreeNode root, ArrayList<ArrayList<Integer>> list, int depth) {
        if (root == null) {
            return;
        }

        if (list.size() == depth) {
            list.add(new ArrayList<>());
        }

        dfs(root.left, list, depth + 1);
        list.get(depth).add(root.val);
        dfs(root.right, list, depth + 1);
    }

    //9对称的二叉树
    boolean isSymmetrical(TreeNode pRoot) {
        return isSame(pRoot, pRoot);
    }

    public boolean isSame(TreeNode root1, TreeNode root2) {
        if (root1 == null && root2 == null) {
            return true;
        }
        if (root1 == null || root2 == null) {
            return false;
        }
        return root1.val == root2.val &&
                isSame(root1.left, root2.right) &&
                isSame(root1.right, root2.left);
    }

    //10最长回文子串
    public int getLongestPalindrome (String A) {
        // write code here
        int maxLen = 0;
        int n = A.length();

        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j <= n; j++) {
                String now = A.substring(i, j);
                if (isHuiWen(now) && now.length() > maxLen) {
                    maxLen = now.length();
                }
            }
        }

        return maxLen;
    }

    public boolean isHuiWen(String s) {
        int len = s.length();
        for (int i = 0; i < len / 2; i++) {
            if (s.charAt(i) != s.charAt(len - i - 1)) {
                return false;
            }
        }
        return true;
    }

    //冒泡排序
    public int[] bubbleSort(int[] nums) {
        for (int i = 0; i < nums.length; i++) {
            for (int j = 0; j < nums.length - i - 1; j++) {
                if (nums[j] > nums[j + 1]) {
                    int temp = nums[j];
                    nums[j] = nums[j + 1];
                    nums[j + 1] = temp;
                }
            }
        }
        return nums;
    }

    //两数之和
    public int[] twoSum(int[] nums, int target) {
        if (Objects.isNull(nums) || nums.length == 0) {
            return null;
        }
        HashMap<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            int complement = target - nums[i];
            if (map.containsKey(complement)) {
                return new int[]{map.get(complement), i};
            }
            map.put(nums[i], i);
        }
        return null;
    }

    //回文数
    public boolean isPalindrome(int x) {
        if (x < 0 || (x % 10 == 0 && x != 0)) {
            return false;
        }
        int revertedNumber = 0;
        while (x > revertedNumber) {
            revertedNumber = revertedNumber * 10 + x % 10;
            x /= 10;
        }
        return x == revertedNumber || x == revertedNumber / 10;
    }

    //======================简单题结束======================

    //======================中等题开始======================
    //======================中等题结束======================
}
